#include "optimiser.h"
#include "mnist_helper.h"
#include "neural_network.h"
#include "math.h"

// Function declarations
void update_parameters(unsigned int batch_size);
void print_training_stats(unsigned int epoch_counter, unsigned int total_iter, double mean_loss, double test_accuracy);

void validate_gradients(unsigned int sample);

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

void run_optimisation(void){
    unsigned int training_sample = 0;
    unsigned int total_iter = 0;
    double obj_func = 0.0;
    unsigned int epoch_counter = 0;
    double test_accuracy = 0.0;  //evaluate_testing_accuracy();
    double mean_loss = 0.0;
    
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

	// Validate gradients (expensive, evaluate infrequently)
	for (int v = 0; v < 10; ++v) {
		validate_gradients(v);
	}
}


void validate_gradients(unsigned int sample){
	// Forward pass
	double loss = evaluate_objective_function(sample);

	// Compute gradients using finite differences
	double epsilon = 0.0001;
	double fd_grad = 0.0;
	double bp_grad = 0.0;
	double diff = 0.0;
	double rel_diff = 0.0;
	double max_rel_diff = 0.0;
	double max_diff = 0.0;
	unsigned int max_diff_i = 0;
	unsigned int max_diff_j = 0;
	unsigned int max_rel_diff_i = 0;
	unsigned int max_rel_diff_j = 0;

	// Validate gradients
	for (int i = 0; i < N_NEURONS_L3; i++){
		for (int j = 0; j < N_NEURONS_LO; j++){
			// Compute gradient using finite differences
			w_L3_LO[i][j].w += epsilon;
			double perturbed_loss = evaluate_objective_function(sample);

			fd_grad = (perturbed_loss - loss)/epsilon;

			// Compute gradient using back-propagation
			bp_grad = w_L3_LO[i][j].dw;

			// Compute difference between gradients
			diff = fabs(fd_grad - bp_grad);
			rel_diff = diff/fabs(fd_grad);

			// Update max diff
			if (diff > max_diff){
				max_diff = diff;
				max_diff_i = i;
				max_diff_j = j;
			}

			// Update max relative diff
			if (rel_diff > max_rel_diff){
				max_rel_diff = rel_diff;
				max_rel_diff_i = i;
				max_rel_diff_j = j;
			}
		}
	}

	// Print max diff
	printf("Max diff: %f, i: %u, j: %u\n", max_diff, max_diff_i, max_diff_j);
	printf("Max rel diff: %f, i: %u, j: %u\n", max_rel_diff, max_rel_diff_i, max_rel_diff_j);
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


