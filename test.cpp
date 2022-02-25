#include <stdlib.h>
#include <stdio.h>
#include <ctime>
#include <chrono>
#include <iostream>
#include "tensorflow/c/c_api.h"

void NoOpDeallocator(void* data, size_t a, void* b) {}
int main_test(int loop_time)
{
    //********* Read model
    TF_Graph* Graph = TF_NewGraph();
    TF_Status* Status = TF_NewStatus();
    TF_SessionOptions* SessionOpts = TF_NewSessionOptions();
    TF_Buffer* RunOpts = NULL;
    
    const char* saved_model_dir = "../web-trace"; 
    const char* tags = "serve"; 
    
    int ntags = 1;
    TF_Session* Session = TF_LoadSessionFromSavedModel(SessionOpts, RunOpts, saved_model_dir, &tags, ntags, Graph, NULL, Status);
    
    if(TF_GetCode(Status) == TF_OK)
    {
        printf("TF_LoadSessionFromSavedModel OK\n");
    }
    else
    {
        printf("%s",TF_Message(Status));
    }

    //****** Get input tensor
    int NumInputs = 1;
    TF_Output* Input = (TF_Output*)malloc(sizeof(TF_Output) * NumInputs);
    TF_Output t0 = {TF_GraphOperationByName(Graph, "serving_default_input_1"), 0};
    
    if(t0.oper == NULL) {
        printf("ERROR: Failed TF_GraphOperationByName serving_default_input_1\n");
	}
    else {
        printf("TF_GraphOperationByName serving_default_input_1 is OK\n");
	}

    Input[0] = t0;
    
    //********* Get Output tensor
    int NumOutputs = 1;
    TF_Output* Output = (TF_Output*)malloc(sizeof(TF_Output) * NumOutputs);
    TF_Output t2 = {TF_GraphOperationByName(Graph, "StatefulPartitionedCall"), 0};
    
    if(t2.oper == NULL) {
        printf("ERROR: Failed TF_GraphOperationByName StatefulPartitionedCall\n");
	}
    else {
		printf("TF_GraphOperationByName StatefulPartitionedCall is OK\n");
	}
    
    Output[0] = t2;

    //********* Allocate data for inputs & outputs
    TF_Tensor** InputValues  = (TF_Tensor**)malloc(sizeof(TF_Tensor*)*NumInputs);
    TF_Tensor** OutputValues = (TF_Tensor**)malloc(sizeof(TF_Tensor*)*NumOutputs);
    
    int ndims = 3;
    int64_t dims[] = {1, 48, 2258};
    //int64_t data[] = {20};
	float data[48 * 2258];

	for (int i = 0; i < 48 * 2258; i++) {
		data[i] = 0.8;
	}
    
    int ndata = sizeof(float) * 48 * 2258; 
    TF_Tensor* int_tensor = TF_NewTensor(TF_FLOAT, dims, ndims, data, ndata, &NoOpDeallocator, 0);
    
    if (int_tensor != NULL)
        printf("TF_NewTensor is OK\n");
    else
      printf("ERROR: Failed TF_NewTensor\n");
    
    InputValues[0] = int_tensor;

	clock_t start = clock();
	auto t_start = std::chrono::high_resolution_clock::now();
    // Run the Session
	//while(1) {
    TF_SessionRun(Session, NULL, Input, InputValues, NumInputs, Output, OutputValues, NumOutputs, NULL, 0, NULL, Status);
	auto t_middle = std::chrono::high_resolution_clock::now();
	clock_t middle = clock();
	for (int i = 0; i < loop_time; i++) {
    TF_SessionRun(Session, NULL, Input, InputValues, NumInputs, Output, OutputValues, NumOutputs, NULL, 0, NULL, Status);
	}

    if(TF_GetCode(Status) == TF_OK)
      printf("Session is OK\n");
    else
      printf("%s",TF_Message(Status));

	clock_t end = clock();
	auto t_end = std::chrono::high_resolution_clock::now();

	// Get dims of output
	int num_dims = TF_GraphGetTensorNumDims(Graph, t2, Status);
	if(TF_GetCode(Status) == TF_OK)
		printf("GraphGetTensorNumDims is OK, dim is %d\n", num_dims);
	else
		printf("%s",TF_Message(Status));

	int64_t output_dims[2];
	TF_GraphGetTensorShape(Graph, t2, output_dims, num_dims, Status);
	if(TF_GetCode(Status) == TF_OK) {
		printf("GraphGetTensorShape is OK\ndims is:");

		for (int i = 0; i < num_dims; i++) {
			printf(" %d", output_dims[i]);
		}
		printf("\n");
	} else {
		printf("%s", TF_Message(Status));
	}

	// Print the output
	void* buff = TF_TensorData(OutputValues[0]);
    float* offsets = (float*)buff;
	/*
    printf("Result Tensor :");
	for (int i = 0; i < 2258; i++) {
		printf(" %f,", offsets[i]);
	}
	printf("\n");
	*/

    // Free memory
    TF_DeleteGraph(Graph);
    TF_DeleteSession(Session, Status);
    TF_DeleteSessionOptions(SessionOpts);
    TF_DeleteStatus(Status);

	printf("start: %d, end: %d, total time: %lf\n", start, end, (double)(end - start) / CLOCKS_PER_SEC);
	//std::cout << "time passed: " << std::chrono::duration<double, std::milli>(t_end-t_start).count() << " ms\n";
	std::cout << "\033[1;33mtime1: " << std::chrono::duration<double, std::milli>(t_middle-t_start).count() << " ms\033[0m\n";
	std::cout << "\033[1;33mtime2: " << std::chrono::duration<double, std::milli>(t_end - t_middle).count() << " ms\033[0m\n";
	return 0;
}

int main(int argc, char * argv[])
{
	main_test(10);
	main_test(50);
	main_test(1000);
	//main_test(10000);
	return 0;
}
