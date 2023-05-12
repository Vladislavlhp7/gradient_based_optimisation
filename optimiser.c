#include <sys/times.h>
#include <time.h>
#include <stdlib.h>
#include "optimiser.h"
#include "mnist_helper.h"
#include "neural_network.h"
#include "math.h"

// Function declarations
void update_parameters(unsigned int batch_size);
void update_parameters_adagrad(unsigned int batch_size);
void print_training_stats(unsigned int epoch_counter, unsigned int total_iter, double mean_loss, double test_accuracy);
void init_parameters_adagrad();
double validate_gradients(unsigned int sample, FILE *f);

// Optimisation parameters
unsigned int log_freq = 30000; // Compute and print accuracy every log_freq iterations

// Parameters passed from command line arguments
unsigned int num_batches;
unsigned int batch_size;
double learning_rate;
unsigned int total_epochs;
unsigned int forward_differencing = 0;
unsigned int backward_differencing = 0;
unsigned int central_differencing = 0;

unsigned int adaptive_learning_rate = 1;
double learning_rate_0 = 0.01;
double learning_rate_N = 0.001;

unsigned int use_adagrad = 0;

unsigned int use_momentum = 0;
double momentum = 0.9;

double** accumulated_gradients_L3_LO;
double** accumulated_gradients_L2_L3;
double** accumulated_gradients_L1_L2;
double** accumulated_gradients_LI_L1;

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

void free_2d_array(double **array, int rows) {
	for (int i = 0; i < rows; i++) {
		free(array[i]);
	}
	free(array);
}

void run_optimisation(void){
    unsigned int training_sample = 0;
    unsigned int total_iter = 0;
    double obj_func = 0.0;
    unsigned int epoch_counter = 0;
    double test_accuracy = 0.0;  //evaluate_testing_accuracy();
    double mean_loss = 0.0;

	if(adaptive_learning_rate){
		printf("Using adaptive learning rate with lr_0 = %f and lr_N = %f\n", learning_rate_0, learning_rate_N);
	}
	if(use_momentum){
		printf("Using momentum with momentum = %f\n", momentum);
	}
	if(use_adagrad){
		printf("Using AdaGrad\n");
		init_parameters_adagrad();
	}
	else{
		printf("Using SGD\n");
	}

	// make an optimization file based on learning rate and batch size
	char optimization_filename[50], validation_filename[50];
	sprintf(optimization_filename, "data/training_%f_%d%d%d%d.txt", learning_rate, batch_size, adaptive_learning_rate, use_adagrad, use_momentum);
	sprintf(validation_filename, "data/validation_%f_%d%d%d%d.txt", learning_rate, batch_size, adaptive_learning_rate, use_adagrad, use_momentum);
	FILE *train_f = fopen(optimization_filename, "w");
	FILE *valid_f;
	if(forward_differencing || backward_differencing || central_differencing){
		valid_f = fopen(validation_filename, "w");
	}
	clock_t start, end;
    // Run optimiser - update parameters after each mini-batch
	start = clock();
    for (int i=0; i < num_batches; i++){
        for (int j = 0; j < batch_size; j++){
            // Evaluate accuracy on testing set (expensive, evaluate infrequently)
            if (total_iter % log_freq == 0 || total_iter == 0){
                if (total_iter > 0){
                    mean_loss = mean_loss/((double) log_freq);
                }
                
                test_accuracy = evaluate_testing_accuracy();
                print_training_stats(epoch_counter, total_iter, mean_loss, test_accuracy);

				// Save to file
	            printf("%d, %f, %f\n", epoch_counter, mean_loss, test_accuracy);
				fprintf(train_f, "%d, %f, %f\n", epoch_counter, mean_loss, test_accuracy);

                // Reset mean_loss for next reporting period
                mean_loss = 0.0;
            }

            // Evaluate forward pass and calculate gradients
            obj_func = evaluate_objective_function(training_sample);
            mean_loss+=obj_func;

	        // Validate gradients (expensive, evaluate infrequently)
			if(total_iter == batch_size-1){
				validate_gradients(training_sample, valid_f);
			}

            // Update iteration counters (reset at end of training set to allow multiple epochs)
            total_iter++;
            training_sample++;
            // On epoch completion:
            if (training_sample == N_TRAINING_SET){
                training_sample = 0;
                epoch_counter++;
            }
        }
	    if(adaptive_learning_rate) {
		    // adaptive learning rate: ηk = η0(1 − α) + αηN
		    double learning_rate_alpha = (double) epoch_counter / (double) total_epochs;
		    learning_rate = learning_rate_0 * (1 - learning_rate_alpha) + learning_rate_N * learning_rate_alpha;
	    }
        // Update weights on batch completion
	    if(use_adagrad){
		    update_parameters_adagrad(batch_size);
	    }
		else{
	        update_parameters(batch_size);
		}
    }
	end = clock();
	double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
	printf("Time spent: %f\n", time_spent);

	// Print final performance
    test_accuracy = evaluate_testing_accuracy();
    print_training_stats(epoch_counter, total_iter, (mean_loss/((double) log_freq)), test_accuracy);

	fclose(train_f);
	fclose(valid_f);
	free_2d_array(accumulated_gradients_L3_LO, N_NEURONS_L3);
	free_2d_array(accumulated_gradients_L2_L3, N_NEURONS_L2);
	free_2d_array(accumulated_gradients_L1_L2, N_NEURONS_L1);
	free_2d_array(accumulated_gradients_LI_L1, N_NEURONS_LI);
}


/*
 * Validate gradients using finite differences
 *
 * @param sample: index of sample to use for validation
 * @param f: file to write to
 *
 * @return: double, maximum relative difference between gradients
 */
double validate_gradients(unsigned int sample, FILE* f){
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
					fprintf(f, "%f, ", rel_diff);
				}
			}
		}
		fprintf(f, "\n");
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
					fprintf(f, "%f, ", rel_diff);
				}
			}
		}
		fprintf(f, "\n");
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
					fprintf(f, "%f, ", rel_diff);
				}
			}
		}
		fprintf(f, "\n");
		fclose(f);
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
			if(use_momentum){
				double dw_current = momentum * w_I_O[i][j].prev_dw - learning_rate * w_I_O[i][j].dw / (double) batch_size;
				w_I_O[i][j].prev_dw = dw_current;
				w_I_O[i][j].w += dw_current;
			}
			else{
				w_I_O[i][j].w -= learning_rate * w_I_O[i][j].dw / (double) batch_size;
			}
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

/*
 * Evaluate forward pass of network using AdaGrad
 *
 * @param parameters: pointer to parameters to update
 * @param gradients: pointer to gradients to use for update
 * @param accumulated_gradients: pointer to accumulated gradients
 * @param learning_rate: learning rate to use for update
 * @param epsilon: epsilon value to use for update
 * @param total_parameters: total number of parameters
 *
 * @return: void
 */
void adagrad_optimizer_update(weight_struct_t* gradients, double* accumulated_gradients, double learning_rate,
							  double epsilon, int rows, int cols, unsigned int batch_size) {
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			int index = i * cols + j;
			accumulated_gradients[index] += gradients[index].dw * gradients[index].dw / (double) batch_size;
			double adaptive_learning_rate = learning_rate / (sqrt(accumulated_gradients[index]) + epsilon);
			gradients[index].w -= adaptive_learning_rate * gradients[index].dw;
			gradients[index].dw = 0;
		}
	}
}


void update_parameters_adagrad(unsigned int batch_size){
	double epsilon = 1e-8;
	adagrad_optimizer_update((weight_struct_t *) w_L3_LO, (double *) accumulated_gradients_L3_LO, learning_rate, epsilon, N_NEURONS_L3, N_NEURONS_LO, batch_size);
	adagrad_optimizer_update((weight_struct_t *) w_L2_L3, (double *) accumulated_gradients_L2_L3, learning_rate, epsilon, N_NEURONS_L2, N_NEURONS_L3, batch_size);
	adagrad_optimizer_update((weight_struct_t *) w_L1_L2, (double *) accumulated_gradients_L1_L2, learning_rate, epsilon, N_NEURONS_L1, N_NEURONS_L2, batch_size);
	adagrad_optimizer_update((weight_struct_t *) w_LI_L1, (double *) accumulated_gradients_LI_L1, learning_rate, epsilon, N_NEURONS_LI, N_NEURONS_L1, batch_size);
}

void init_parameters_adagrad(){
	double init_value = 0.0;
	accumulated_gradients_L3_LO = (double **)malloc(sizeof(double *) * N_NEURONS_L3);
	accumulated_gradients_L2_L3 = (double **)malloc(sizeof(double *) * N_NEURONS_L2);
	accumulated_gradients_L1_L2 = (double **)malloc(sizeof(double *) * N_NEURONS_L1);
	accumulated_gradients_LI_L1 = (double **)malloc(sizeof(double *) * N_NEURONS_LI);

	for (int i = 0; i < N_NEURONS_L3; i++) {
		accumulated_gradients_L3_LO[i] = (double *)malloc(sizeof(double) * N_NEURONS_LO);
	}

	for (int i = 0; i < N_NEURONS_L2; i++) {
		accumulated_gradients_L2_L3[i] = (double *)malloc(sizeof(double) * N_NEURONS_L3);
	}

	for (int i = 0; i < N_NEURONS_L1; i++) {
		accumulated_gradients_L1_L2[i] = (double *)malloc(sizeof(double) * N_NEURONS_L2);
	}

	for (int i = 0; i < N_NEURONS_LI; i++) {
		accumulated_gradients_LI_L1[i] = (double *)malloc(sizeof(double) * N_NEURONS_L1);
	}

	for(int i = 0; i < N_NEURONS_L3; i++){
		for(int j = 0; j < N_NEURONS_LO; j++){
			accumulated_gradients_L3_LO[i][j] = init_value;
		}
	}
	for(int i = 0; i < N_NEURONS_L2; i++){
		for(int j = 0; j < N_NEURONS_L3; j++){
			accumulated_gradients_L2_L3[i][j] = init_value;
		}
	}
	for(int i = 0; i < N_NEURONS_L1; i++){
		for(int j = 0; j < N_NEURONS_L2; j++){
			accumulated_gradients_L1_L2[i][j] = init_value;
		}
	}
	for(int i = 0; i < N_NEURONS_LI; i++){
		for(int j = 0; j < N_NEURONS_L1; j++){
			accumulated_gradients_LI_L1[i][j] = init_value;
		}
	}
}

