#define THREAD_NUM 8

#include <stdio.h>
#include <pthread.h>

void * thread_func(void * p)
{
	// infinite loop
	while (1);
}

int main(int argc, char * argv[])
{
	pthread_t pool[THREAD_NUM];

	for (int i = 0; i < THREAD_NUM; i++) {
		if (pthread_create(&pool[i], NULL, thread_func, NULL)) {
			printf("Failed to create thread %d\n", i);
			break;
		}
	}

	for (int i = 0; i < THREAD_NUM; i++) {
		pthread_join(pool[i], NULL);
	}

	return 0;
}
