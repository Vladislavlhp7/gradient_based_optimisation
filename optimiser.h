#ifndef OPTIMISER_H
#define OPTIMISER_H

#include <stdio.h>

static const int gradient_validation_num_samples = 1;

void initialise_optimiser(double learning_rate, int batch_size, int total_epochs);
void run_optimisation(void);
double evaluate_objective_function(unsigned int sample);

#endif /* OPTMISER_H */
