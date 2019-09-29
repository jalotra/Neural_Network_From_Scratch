#include <math.h>
#include <string.h>
#include "image.h"
#include "test.h"
#include "args.h"
#include "matrix.h"

int do_test()
{
    TEST('1' == '1');
    TEST('0' == '1');
    return 0;
}

int main(int argc, char **argv)
{
	do_test();
	test_matrix();
    return 0;
}