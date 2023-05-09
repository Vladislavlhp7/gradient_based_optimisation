#include <sys/times.h>
#include <time.h>
#include "optimiser.h"
#include "mnist_helper.h"
#include "neural_network.h"
#include "math.h"

// Function declarations
void update_parameters(unsigned int batch_size);
void print_training_stats(unsigned int epoch_counter, unsigned int total_iter, double mean_loss, double test_accuracy);

double validate_gradients(unsigned int sample);

// Optimisation parameters
unsigned int log_freq = 30000; // Compute and print accuracy every log_freq iterations

// Paramters passed from command line arguments
unsigned int num_batches;
unsigned int batch_size;
unsigned int total_epochs;
unsigned int forward_differencing = 0;
unsigned int backward_differencing = 0;
unsigned int central_differencing = 0;

double learning_rate;

void print_training_stats(unsigned int epoch_counter, unsigned int total_iter, double mean_loss, double test_accuracy){
    printf("Epoch: %u,  Total iter: %u,  Mean Loss: %0.12f,  Test Acc: %f\n", epoch_counter, total_iter, mean_loss, test_accuracy);
}

void initialise_optimiser(double cmd_line_learning_rate, int cmd_line_batch_size, int cmd_line_total_epochs){
    batch_size = cmd_line_batch_size;
    learning_rate = cmd_line_learning_rate;
    total_epochs = cmd_line_total_epochs;
    
    num_batches = total_epochs * (N_TRAINING_SET / batch_size);
    printf("Optimising with paramters: \n\tepochs = %u \n\tbatch_size = %u \n\tnum_batches = %u\n\tlearning_rate = %f\n\n",
           total_epochs, batch_size, num_batches, learning_rate);
}

/*
 * Plot the validation of the gradients over time in a graph
 */
void plot_gradients(void){
	FILE *f = fopen("grad_validation.txt", "r");
	if (f == NULL) {
		printf("Error opening file!\n");
		return;
	}
	char * commandsForGnuplot[] = {
			"set title \"Relative FD Error over Training Epochs\"",
			"set ylabel \"Relative FD Error\"",
			"set xlabel \"Number of epochs\"",
			"plot 'grad_validation.txt' using 1:2 with lines dt 2 lt 9 lc 7 title 'Absolute Error - Provided SR', 'part1.temp' using 1:3 with lines lc 6 title 'Absolute Error - Alternative SR'"
	};
	FILE * gnuplotPipe = popen ("gnuplot -persistent", "w");
	for (int i=0; i < 4; i++){
		fprintf(gnuplotPipe, "%s \n", commandsForGnuplot[i]); //Send commands to gnuplot one by one.
	}
}

void run_optimisation(void){
    unsigned int training_sample = 0;
    unsigned int total_iter = 0;
    double obj_func = 0.0;
    unsigned int epoch_counter = 0;
    double test_accuracy = 0.0;  //evaluate_testing_accuracy();
    double mean_loss = 0.0;

	// Validate gradients (expensive, evaluate infrequently)
	double grad_valid_arr[gradient_validation_num_samples];

    // Run optimiser - update parameters after each mini-batch
    for (int i=0; i < num_batches; i++){
        for (int j = 0; j < batch_size; j++){

            // Evaluate accuracy on testing set (expensive, evaluate infrequently)
            if (total_iter % log_freq == 0 || total_iter == 0){
                if (total_iter > 0){
                    mean_loss = mean_loss/((double) log_freq);
                }
                
                test_accuracy = evaluate_testing_accuracy();
                print_training_stats(epoch_counter, total_iter, mean_loss, test_accuracy);

                // Reset mean_loss for next reporting period
                mean_loss = 0.0;
            }

            // Evaluate forward pass and calculate gradients
            obj_func = evaluate_objective_function(training_sample);
            mean_loss+=obj_func;

	        // Validate gradients (expensive, evaluate infrequently)
	        validate_gradients(training_sample);

            // Update iteration counters (reset at end of training set to allow multiple epochs)
            total_iter++;
            training_sample++;
            // On epoch completion:
            if (training_sample == N_TRAINING_SET){
                training_sample = 0;
                epoch_counter++;
            }
        }

        // Update weights on batch completion
        update_parameters(batch_size);
    }
    // Print final performance
    test_accuracy = evaluate_testing_accuracy();
    print_training_stats(epoch_counter, total_iter, (mean_loss/((double) log_freq)), test_accuracy);


//	FILE *f = fopen("grad_validation.txt", "a");
//	for (int v = 0; v < gradient_validation_num_samples; ++v) {
//		// Validate gradients (expensive, evaluate infrequently)
//		grad_valid_arr[v] = validate_gradients(v);
//		fprintf(f, "%d, %d, %f\n", total_epochs, v, grad_valid_arr[v]);
//	}
//	fclose(f);
	// Plot gradients
	// plot_gradients();
}


/*
 * Validate gradients using finite differences
 *
 * @param sample: index of sample to use for validation
 *
 * @return: double, maximum relative difference between gradients
 */
double validate_gradients(unsigned int sample){
	// Compute gradients using finite differences
	// set epsilon to 10e-8
	double epsilon = 0.00000001;
	double diff_accumulated = 0.0;
	double rel_diff_accumulated = 0.0;
	double time_spent, avg_diff = 0, avg_rel_diff;
	clock_t start, end;
	int rel_iter = 0;

	// Validate gradients using forward, backward and central difference
	if(forward_differencing){
		start = clock();
		// Validate gradients using forward difference
		for (int i = 0; i < N_NEURONS_L3; i++) {
			for (int j = 0; j < N_NEURONS_LO; j++) {
				// Compute gradient using forward difference
				w_L3_LO[i][j].w += epsilon;
				evaluate_forward_pass(training_data, sample);
				double perturbed_loss_plus_eps = compute_xent_loss(training_labels [sample]);

				w_L3_LO[i][j].w -= epsilon;
				evaluate_forward_pass(training_data, sample);
				double perturbed_loss = compute_xent_loss(training_labels [sample]);

				double numerical_grad = (perturbed_loss_plus_eps - perturbed_loss) / epsilon;
				double analytical_grad = dL_dW_L3_LO[0][i+(N_NEURONS_L3 * j)];

				// Compute difference between gradients
				double diff = fabs(numerical_grad - analytical_grad);
				double rel_diff = (diff / fabs(analytical_grad)) * 100.0;
				diff_accumulated += diff;
				if(analytical_grad > 0.0){
					rel_iter++;
					rel_diff_accumulated += rel_diff;
				}
			}
		}
		end = clock();
		time_spent = (double)(end - start) / CLOCKS_PER_SEC;
		avg_diff = diff_accumulated / (N_NEURONS_L3 * N_NEURONS_LO);
		avg_rel_diff = rel_diff_accumulated / rel_iter;
		printf("Forward Diff: Average diff: %.32f, Percentage avg rel_diff: %.32f, Time: %.5f\n", avg_diff, avg_rel_diff, time_spent);
	}

	if(backward_differencing){
		diff_accumulated = 0.0;
		rel_diff_accumulated = 0.0;
		rel_iter = 0;
		start = clock();
		// Validate gradients using backward difference
		for (int i = 0; i < N_NEURONS_L3; i++) {
			for (int j = 0; j < N_NEURONS_LO; j++) {
				// Compute gradient using backward difference
				w_L3_LO[i][j].w -= epsilon;
				evaluate_forward_pass(training_data, sample);
				double perturbed_loss_minus_eps = compute_xent_loss(training_labels [sample]);

				w_L3_LO[i][j].w += epsilon;
				evaluate_forward_pass(training_data, sample);
				double perturbed_loss = compute_xent_loss(training_labels [sample]);

				double numerical_grad = (perturbed_loss - perturbed_loss_minus_eps) / epsilon;
				double analytical_grad = dL_dW_L3_LO[0][i+(N_NEURONS_L3 * j)];

				// Compute difference between gradients
				double diff = fabs(numerical_grad - analytical_grad);
				double rel_diff = (diff / fabs(analytical_grad)) * 100.0;
				diff_accumulated += diff;
				if(analytical_grad > 0.0){
					rel_iter++;
					rel_diff_accumulated += rel_diff;
				}
			}
		}
		end = clock();
		time_spent = (double)(end - start) / CLOCKS_PER_SEC;
		avg_diff = diff_accumulated / (N_NEURONS_L3 * N_NEURONS_LO);
		avg_rel_diff = rel_diff_accumulated / rel_iter;
		printf("Backward Diff: Average diff: %.32f, Percentage avg rel_diff: %.32f, Time: %.5f\n", avg_diff, avg_rel_diff, time_spent);
	}

	if(central_differencing){
		diff_accumulated = 0.0;
		rel_diff_accumulated = 0.0;
		rel_iter = 0;
		start = clock();
		// Validate gradients using central difference
		for (int i = 0; i < N_NEURONS_L3; i++) {
			for (int j = 0; j < N_NEURONS_LO; j++) {
				w_L3_LO[i][j].w += epsilon;
				evaluate_forward_pass(training_data, sample);
				double perturbed_loss_plus_eps = compute_xent_loss(training_labels [sample]);

				w_L3_LO[i][j].w -= 2 * epsilon;
				evaluate_forward_pass(training_data, sample);
				double perturbed_loss_minus_eps = compute_xent_loss(training_labels [sample]);

				w_L3_LO[i][j].w += epsilon;

				double numerical_grad = (perturbed_loss_plus_eps - perturbed_loss_minus_eps) / (2 * epsilon);
				double analytical_grad = dL_dW_L3_LO[0][i+(N_NEURONS_L3 * j)];

				// Compute difference between gradients
				double diff = fabs(numerical_grad - analytical_grad);
				double rel_diff = (diff / fabs(analytical_grad)) * 100;

				diff_accumulated += diff;
				if(analytical_grad > 0.0){
					rel_iter++;
					rel_diff_accumulated += rel_diff;
				}
			}
		}
		end = clock();
		time_spent = (double)(end - start) / CLOCKS_PER_SEC;
		avg_diff = diff_accumulated / (N_NEURONS_L3 * N_NEURONS_LO);
		avg_rel_diff = rel_diff_accumulated / rel_iter;
		printf("Central Diff: Average diff: %.32f, Percentage avg rel_diff: %.32f, Time: %.5f\n", avg_diff, avg_rel_diff, time_spent);
	}
	return avg_diff;
}

double evaluate_objective_function(unsigned int sample){

    // Compute network performance
    evaluate_forward_pass(training_data, sample);
    double loss = compute_xent_loss(training_labels[sample]);
    
    // Evaluate gradients
    //evaluate_backward_pass(training_labels[sample], sample);
    evaluate_backward_pass_sparse(training_labels[sample], sample);
    
    // Evaluate parameter updates
    store_gradient_contributions();
    
    return loss;
}

/*
 * Evaluate forward pass of network
 *
 * @param data: pointer to training data
 * @param sample: index of sample to use for forward pass
 *
 * @return: void
 */
void update_weights(unsigned int N_NEURONS_I, unsigned int N_NEURONS_O,
					weight_struct_t w_I_O[N_NEURONS_I][N_NEURONS_O], unsigned int batch_size){
	// Update weights for given layers using mini-batch gradient descent
	for (int i = 0; i < N_NEURONS_I; ++i) {
		for (int j = 0; j < N_NEURONS_O; ++j) {
			w_I_O[i][j].w -= learning_rate * w_I_O[i][j].dw / (double) batch_size;
			w_I_O[i][j].dw = 0;
		}
	}
}

void update_parameters(unsigned int batch_size){
    // Part I To-do
	update_weights(N_NEURONS_L3, N_NEURONS_LO, w_L3_LO, batch_size);
	update_weights(N_NEURONS_L2, N_NEURONS_L3, w_L2_L3, batch_size);
	update_weights(N_NEURONS_L1, N_NEURONS_L2, w_L1_L2, batch_size);
	update_weights(N_NEURONS_LI, N_NEURONS_L1, w_LI_L1, batch_size);
}


