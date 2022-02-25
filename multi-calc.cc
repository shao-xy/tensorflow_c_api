#include "tf/TFContext.h"

#define MODEL_DIR "../web-trace"
#define ROUND 10

int main(int argc, char * argv[])
{
	TFContext ctx;
	ctx.load_model(MODEL_DIR);
	ctx.run_test(ROUND);
	return 0;
}
