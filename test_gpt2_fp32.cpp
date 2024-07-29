#define TESTING
#include "train_gpt2_fp32.cpp"

// poor man's tensor checker
int check_tensor(float *a, float *b, int n, const char* label) {
    int print_upto = 5;
    bool ok = true;
    printf("%s\n", label);
    for (int i = 0; i < n; i++) {
        if (fabsf(a[i] - b[i]) <= 1e-2) {
            if (i < print_upto) { printf("OK "); }
        } else {
            if (i < print_upto) { printf("NOT OK "); }
            ok = false;
        }
        if (i < print_upto) { printf("%f %f\n", a[i], b[i]); }
    }
    // print the final result
    if (ok) {
        printf("TENSOR OK\n");
    } else {
        printf("TENSOR NOT OK\n");
    }
    return ok;
}

int main(int argc, char *argv[]) {

    // set up the device
    sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());

    std::cout << "[System]\n";
    std::cout << "Device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str();

    // build the GPT-2 model from a checkpoint
    GPT2 model;
    gpt2_build_from_checkpoint(q, &model, "gpt2_124M.bin");

    // int C = model.config.channels;
    int V = model.config.vocab_size;
    int Vp = model.config.padded_vocab_size;
    int maxT = model.config.max_seq_len;
    // int L = model.config.num_layers;

    // load additional information that we will use for debugging and error checking
    FILE *state_file = fopenCheck("gpt2_124M_debug_state.bin", "rb");
    int state_header[256];
    freadCheck(state_header, sizeof(int), 256, state_file);
    if (state_header[0] != 20240327) { printf("Bad magic state file\n"); exit(EXIT_FAILURE); }
    if (state_header[1] != 2) {
        fprintf(stderr, "Bad version in state file\n");
        fprintf(stderr, "---> HINT: try to re-run `python train_gpt2.py`\n");
        exit(EXIT_FAILURE);
    }
    int B = state_header[2]; // batch size, e.g. 4
    int T = state_header[3]; // time / sequence length (e.g. 64, up to maxT)
    assert(0 <= T && T <= maxT);
    std::cout << "[State]\n";
    std::cout << "batch_size: " << B << "\n";
    std::cout << "seq_len: " << T << "\n";

    ParameterTensors expected_grads; // will be read from file (from PyTorch)
    ParameterTensors calculated_grads; // will be calculated by us
    float* expected_grads_memory = malloc_and_point_parameters(q, &expected_grads, model.param_sizes, 0);
    float* calculated_grads_memory = malloc_and_point_parameters(q, &calculated_grads, model.param_sizes, 0);

    // inputs and expected outputs, only used for error checking
    int* x = (int*)mallocCheck(B * T * sizeof(int));
    int* y = (int*)mallocCheck(B * T * sizeof(int));
    float* expected_logits = (float*) mallocCheck(B * T * V * sizeof(float));
    float* expected_loss = (float*) mallocCheck(1 * sizeof(float));

    // read reference information from Python
    freadCheck(x, sizeof(int), B*T, state_file);
    freadCheck(y, sizeof(int), B*T, state_file);
    freadCheck(expected_logits, sizeof(float), B*T*V, state_file);
    freadCheck(expected_loss, sizeof(float), 1, state_file);
    freadCheck(expected_grads_memory, sizeof(float), model.num_parameters, state_file);
    fcloseCheck(state_file);

    // overall OK signal for the test
    bool allok = true;

    // First, do target-free forward pass to validate logits
    gpt2_forward(&model, x, nullptr, B, T, q);
    // at this point, target should be equal to expected_logits, let's compare
    // copy logits to CPU so we can compare them
    float* logits_cpu = (float*)mallocCheck(B * T * Vp * sizeof(float));
    q.memcpy(logits_cpu, model.acts.output, B * T * Vp * sizeof(float)).wait();

    // compare the output logits from the forward pass
    // also careful that we don't access and compare the padded columns of logits
    bool logits_ok = true;
    float max_diff = 0.0f;
    for (int bt = 0; bt < B*T; bt++) {
        for (int v = 0; v < V; v++) {
            int i = bt * Vp + v; // linearized index
            if (i < 10) {
                printf("%f, %f\n", expected_logits[i], logits_cpu[i]);
            }
            float diff = fabsf(expected_logits[bt*V + v] - logits_cpu[i]);
            max_diff = fmaxf(max_diff, diff);
            if (diff >= 1e-2f) {
/*                printf("MISMATCH AT INDEX %d,%d: ", bt, v);
                printf("%f %f\n", expected_logits[bt*V + v], logits_cpu[i]);*/
                std::cout << "MISMATCH AT INDEX " << bt << "," << v << ": "
                          << expected_logits[bt*V + v] << " " << logits_cpu[i] << std::endl;
                logits_ok = false;
                bt = B*T; // to break out of both loops
                break;
            }
        }
    }
    allok = allok && logits_ok;
    if(!logits_ok) { printf("NOT "); }
    printf("OK (LOGITS)\n");

    // let's do 10 training iterations, following the pytorch code
    float losses[10];
    for (int step = 0; step < 10; step++) {
        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);
        gpt2_forward(&model, x, y, B, T, q);
        gpt2_zero_grad(&model, q);
        gpt2_backward(&model, q);
        clock_gettime(CLOCK_MONOTONIC, &end);
        double time_elapsed_s = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

        if (step == 0) {
            // error checking at step 0 for reference activations
            free(logits_cpu);

            // compare the achieved loss
            if (fabsf(model.mean_loss - *expected_loss) >= 1e-2) {
                printf("LOSS MISMATCH: %f %f\n", model.mean_loss, *expected_loss);
                allok = false;
            } else {
                printf("LOSS OK: %f %f\n", model.mean_loss, *expected_loss);
            }

            // compare the gradients ona the parameters all at once
            q.memcpy(calculated_grads_memory, model.grads_memory, model.num_parameters * sizeof(float)).wait();
            check_tensor(calculated_grads_memory, expected_grads_memory, model.num_parameters, "grads");
        }
        gpt2_update(&model, 1e-4f, 0.9f, 0.999f, 1e-8f, 0.01f, step+1, q);

        // print the timing information at the end
        printf("step %d: loss %f (took %f ms)\n", step, model.mean_loss, time_elapsed_s * 1000);
        losses[step] = model.mean_loss;
    }

    // expected losses are as follows, from Python
    float expected_losses[10] = {
            5.270007133483887,
            4.059706687927246,
            3.3751230239868164,
            2.8007826805114746,
            2.315382242202759,
            1.8490285873413086,
            1.3946564197540283,
            0.9991465210914612,
            0.6240804195404053,
            0.37651097774505615
    };

    // compare
    for (int i = 0; i < 10; i++) {
        if (fabsf(losses[i] - expected_losses[i]) >= 1e-2) {
            printf("LOSS MISMATCH AT STEP %d: %f %f\n", i, losses[i], expected_losses[i]);
            allok = false;
        } else {
            printf("loss ok at step %d: %f %f\n", i, losses[i], expected_losses[i]);
        }
    }

    // final approval
    std::cout << "overall okay: " << allok << std::endl;

    // free everything
    free(x);
    free(y);
    free(expected_logits);
    free(expected_loss);
    free(expected_grads_memory);
    free(calculated_grads_memory);
    gpt2_free(&model, q);

    return 0;
}