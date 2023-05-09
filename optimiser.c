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


	FILE *f = fopen("grad_validation.txt", "a");
	for (int v = 0; v < gradient_validation_num_samples; ++v) {
		// Validate gradients (expensive, evaluate infrequently)
		grad_valid_arr[v] = validate_gradients(v);
		fprintf(f, "%d, %d, %f\n", total_epochs, v, grad_valid_arr[v]);
	}
	fclose(f);
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
	double fd_grad = 0.0;
	double bp_grad = 0.0;
	double diff = 0.0;
	double rel_diff = 0.0;
	// avg difference between gradients
	double avg_diffs[gradient_validation_num_samples];
	double diff_accumulated = 0.0;
	double rel_diff_accumulated = 0.0;

	 clock_t start = clock();

	// Validate gradients
	for (int i = 0; i < N_NEURONS_L3; i++){
		for (int j = 0; j < N_NEURONS_LO; j++){
			// Compute gradient using finite differences
			w_L3_LO[i][j].w += epsilon;
			double perturbed_loss_1 = evaluate_objective_function(sample);
			w_L3_LO[i][j].w -= 2 * epsilon;
			double perturbed_loss_2 = evaluate_objective_function(sample);

			fd_grad = (perturbed_loss_1 - perturbed_loss_2)/(2 * epsilon);

			// Compute gradient using back-propagation
			bp_grad = w_L3_LO[i][j].dw;

			// Compute average difference between gradients
			diff = fabs(fd_grad - bp_grad);
			rel_diff = diff/fabs(fd_grad);

			// Accumulate differences
			diff_accumulated += diff;
			rel_diff_accumulated += rel_diff;

			// Reset the weight back to its original value
			w_L3_LO[i][j].w -= 2 * epsilon;
		}
	}
	clock_t end = clock();

	// Compute average difference between gradients
	avg_diffs[0] = diff_accumulated/((double) (N_NEURONS_L3*N_NEURONS_LO));
	printf("Avg abs diff: %f \t Time taken: %f\n", avg_diffs[0], ((double) (end - start)) / CLOCKS_PER_SEC);
	return avg_diffs[0];
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
void update_weights(unsigned int N_NEURONS_I, unsigned int N_NEURONS_O, weight_struct_t w_I_O[N_NEURONS_I][N_NEURONS_O]){
	// Update weights for given layers using mini-batch gradient descent
	for (int i = 0; i < N_NEURONS_I; ++i) {
		for (int j = 0; j < N_NEURONS_O; ++j) {
			w_I_O[i][j].w -= learning_rate * w_I_O[i][j].dw;
			w_I_O[i][j].dw = 0;
		}
	}
}

void update_parameters(unsigned int batch_size){
    // Part I To-do
	update_weights(N_NEURONS_L3, N_NEURONS_LO, w_L3_LO);
	update_weights(N_NEURONS_L2, N_NEURONS_L3, w_L2_L3);
	update_weights(N_NEURONS_L1, N_NEURONS_L2, w_L1_L2);
	update_weights(N_NEURONS_LI, N_NEURONS_L1, w_LI_L1);
}


