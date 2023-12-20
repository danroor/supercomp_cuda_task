#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>
#include <string.h>

// уравнение левой боковой стороны трапеции
double trap_left_y(double x) {
	return 3 * x + 9;
}
// x по y
double trap_left_x(double y) {
	return (y - 9) / 3;
}

// уравнение правой боковой стороны трапеции
double trap_right_y(double x) {
	return -3 * x + 9;
}
// x по y
double trap_right_x(double y) {
	return (9 - y) / 3;
}


int inside_trap(double x, double y) {

	return  (y >= 0) && (y <= 3) && 
	        (y <= trap_left_y(x)) && 
	        (y <= trap_right_y(x));

}

// коэффициенты а из формулы 11
void free_a(double **a, int M, int N) {
	for (int i = 1; i <= M; ++i) {
		free(a[i]);
	}
	free(a);
}

double **a_cmp(double *w1, double *w2, int M, int N, double h1, double h2) {
	// добавим фиктивные строку и столбец с индексом 0 во избежание путаницы в индексации
	double **a = calloc(M + 1, sizeof(double *));

	double x_m, y_m, y_p, L, y_intersec;
	int P1_inside, P2_inside;

	double eps = h1 * h1;
	if (h2 > h1) {
		eps = h2 * h2;
	}

	for (int i = 1; i <= M; ++i) {
		a[i] = calloc(N, sizeof(double));

		for (int j = 1; j <= N - 1; ++j) {
			// x_{i-1/2}
			x_m = w1[i] - h1 / 2;
			// y_{j-1/2}
			y_m = w2[j] - h2 / 2;
			// y_{j+1/2}
			y_p = w2[j] + h2 / 2;

			P1_inside = inside_trap(x_m, y_m);
			P2_inside = inside_trap(x_m, y_p);


			if (P2_inside) {
				// верхняя точка внутри трапеции => нижняя тоже
				L = h2;
			} else if (!P1_inside) {
				// отрезок целиком снаружи
				L = 0;
			} else if (x_m < -2) {
				// пересечение с левой боковой стороной
				y_intersec = trap_left_y(x_m);
				L = y_intersec - y_m;
			} else if (x_m > 2) {
				// пересечение с правой боковой стороной
				y_intersec = trap_right_y(x_m);
				L = y_intersec - y_m;
			}

			a[i][j] = (L + (h2 - L) / eps ) / h2;
		}
	
	}

	return a;

}

// коэффициенты b из формулы 11
void free_b(double **b, int M, int N) {
	for (int i = 1; i <= M - 1; ++i) {
			free(b[i]);
	}
	free(b);
}

double **b_cmp(double *w1, double *w2, int M, int N, double h1, double h2) {
	// добавим фиктивные строку и столбец с индексом 0 во избежание путаницы
	double **b = calloc(M, sizeof(double *));

	double y_m, x_m, x_p, L, x_intersec;
	int P1_inside, P2_inside;

	double eps = h1 * h1;
	if (h2 > h1) {
		eps = h2 * h2;
	}

	for (int i = 1; i <= M - 1; ++i) {
		b[i] = calloc(N + 1, sizeof(double));

		for (int j = 1; j <= N; ++j) {
			// y_{j-1/2}
			y_m = w2[j] - h2 / 2;
			// x_{i-1/2}
			x_m = w1[i] - h1 / 2;
			// x_{i+1/2}
			x_p = w1[i] + h1 / 2;

			P1_inside = inside_trap(x_m, y_m);
			P2_inside = inside_trap(x_p, y_m);


			if (P1_inside) {
				if (P2_inside) {
					// отрезок целиком внутри
					L = h1;
				} else {
					// правая точка снаружи, левая внутри
					x_intersec = trap_right_x(y_m);
					L = x_intersec - x_m;
				}
			} else if (!P2_inside) {
				// отрезок целиком снаружи
				L = 0;
			} else {
				//левая точка снаружи, правая внутри
				x_intersec = trap_left_x(y_m);
				L = x_p - x_intersec;
			}

			b[i][j] = (L + (h1 - L) / eps ) / h1;
		}
	
	}

	return b;

}

// правая часть F разностной схемы
void free_F(double **F, int M, int N) {
	for (int i = 1; i <= M - 1; ++i) {
			free(F[i]);
	}
	free(F);
}

double **F_cmp(double *w1, double *w2, int M, int N, double h1, double h2) {

	double **F = calloc(M, sizeof(double *));

	double x_m, y_m, x_p, y_p;
	double y_x_m, y_x_p, x_y_p, x_y_m;

	// площадь каждого прямоугольника П_ij
	double S = h1 * h2;

	for (int i = 1; i <= M - 1; ++i) {
		F[i] = calloc(N, sizeof(double));
		for (int j = 1; j <= N - 1; ++j) {
			// x_{i-1/2}
			x_m = w1[i] - h1 / 2;
			// x_{i+1/2}
			x_p = w1[i] + h1 / 2;
			// y_{j-1/2}
			y_m = w2[j] - h2 / 2;
			// y_{j+1/2}
			y_p = w2[j] + h2 / 2;


			if (x_m * x_p <= 0) {
				// разные знаки координат или равенство одной из них нулю
				// при достаточно больших M, N (не менее 10) означают,
				// что прямоугольник П_ij в центре трапеции, далеко от 
				// ее боковых сторон, и полностью попадает внутрь трапеции
				F[i][j] = S;

			} else if (x_p > 0) {
				// в силу симметрии трапеции относительно оси х 
				// при зеркальном отражении прямоугольника из  
				// положительной полуплоскости относительно оси х
				// получим ту же площадь фигуры в интеграле,
				// что и в случае отрицательных координат по оси х
				y_x_m = x_m;
				x_m = -x_p;
				x_p = -y_x_m;
			}

			y_x_m = trap_left_y(x_m);
			y_x_p = trap_left_y(x_p);
			x_y_m = trap_left_x(y_m);
			x_y_p = trap_left_x(y_p);

			if (y_x_m >= y_p) {
				// Пij целиком внутри трапеции
				F[i][j] = S;
			} else if (y_x_m >= y_m) {
				if (y_x_p >= y_p) {
					// отрезаем треугольник
					F[i][j] = S - (y_p - y_x_m) * (x_y_p - x_m) / 2;
				} else if (y_x_m == y_m) {
					// отрезаем трапецию, справа остаётся треугольник
					F[i][j] = h1 * (y_x_p - y_m) / 2;
				} else {
					// справа остаётся трапеция
					F[i][j] = h1 * (y_x_m - y_m) + 
					          h1 * (y_x_p - y_x_m) / 2;
				}
			} else if (y_x_p > y_p) {
				// справа остаётся трапеция
				F[i][j] = h2 * (x_p - x_y_p) + 
				          h2 * (x_y_p - x_y_m) / 2;
			} else if (y_x_p > y_m) {
				// справа остаётся треугольник
				F[i][j] = (y_x_p - y_m) * (x_p - x_y_m) / 2;
			} else {
				// целиком снаружи
				F[i][j] = 0;
			}

			F[i][j] /= h1 * h2;
		}

	}

	return F;
}

void Aw_cmp(
	// указатель на выделенную под результат память
	double **Aw,
	// сеточная функция, к которой применяется оператор
	double **w,
	double **a, double **b, int M, int N, double h1, double h2) {	

	for (int i = 1; i <= M - 1; ++i) {
		for (int j = 1; j <= N - 1; ++j) {
			Aw[i][j] = -( a[i+1][j] * (w[i+1][j] - w[i][j]) - 
				          a[i][j] *   (w[i][j]   - w[i-1][j]) ) / (h1*h1)

			           -( b[i][j+1] * (w[i][j+1] - w[i][j]) - 
				          b[i][j] *   (w[i][j]   - w[i][j-1]) ) / (h2*h2);
		}
	}
}

void matr_diff(double **dst, double **m1, double **m2, int M, int N) {
	for (int i = 1; i <= M - 1; ++i) {
		for (int j = 1; j <= N - 1; ++j) {
			dst[i][j] = m1[i][j] - m2[i][j];
		}
	}
}

double dot(double **u, double **v, int M, int N, double h1, double h2) {
	double uv = 0;
	for (int i = 1; i <= M - 1; ++i) {
		for (int j = 1; j <= N - 1; ++j) {
			uv += u[i][j] * v[i][j];
		}
	}

	return h1 * h2 * uv;
}

double sqnorm(double **u, int M, int N, double h1, double h2) {
	return dot(u, u, M, N, h1, h2);
}

void print_net_function(double **net, double *w1, double *w2, int M, int N) {
	printf("%15.10f ", -1.0);

	for (int i = 0; i <= M; ++i) {
		printf("%15.10f ", w1[i]);
	}

	printf("\n");

	for (int j = 0; j <= N; ++j) {
		printf("%15.10f ", w2[j]);
		for (int i = 0; i <= M; ++i) {
			printf("%15.10f ", net[i][j]);
		}
		printf("\n");
	}	
}

int main(int argc, char *argv[])
{

	// инициализация работы программы
	struct timeval start, end;
	gettimeofday(&start, NULL);

	if (argc < 3) {
		printf("Неверные параметры сетки\n");
		return 0;
	}

	int M = 0, N = 0, digit, lenM = strlen(argv[1]), lenN = strlen(argv[2]);

	for (int i = 0; i < lenM; ++i) {
		// char digit to one-digit number
		digit = argv[1][i] - '0';
		if (digit < 0 || digit > 9) {
			printf("Неверные параметры сетки\n");
			return 0;		
		}

		M = M * 10 + digit;
	}

	for (int i = 0; i < lenN; ++i) {
		// char digit to one-digit number
		digit = argv[2][i] - '0';
		if (digit < 0 || digit > 9) {
			printf("Неверные параметры сетки\n");
			return 0;		
		}

		N = N * 10 + digit;
	}	

	int print_result = 0, print_loop_cpy_exch_time = 0;

	if (argc > 3 && strlen(argv[3]) == 1 && argv[3][0] == '1') {
		print_result = 1;
	}

	if (argc > 4 && strlen(argv[4]) == 1 && argv[4][0] == '1') {
		print_loop_cpy_exch_time = 1;
	}

	printf("M = %d, N = %d\n", M, N);

	// прямоугольник 
	double A1 = -3, B1 = 3, A2 = 0, B2 = 3;

	double h1 = (B1 - A1) / M;
	double h2 = (B2 - A2) / N;

	double *w1 = calloc(M + 1, sizeof(double));
	double *w2 = calloc(N + 1, sizeof(double));

	double **w_prev = calloc(M + 1, sizeof(double *));
	double **r = calloc(M + 1, sizeof(double *));
	double **Ar = calloc(M + 1, sizeof(double *));

	for (int i = 0; i <= M; ++i) {
		
		w_prev[i] = calloc(N + 1, sizeof(double));
		r[i] = calloc(N, sizeof(double));
		Ar[i] = calloc(N, sizeof(double));

		// сетка на прямоугольнике
		w1[i] = A1 + i * h1;
		
		for (int j = 0; j <= N; ++j) {
			// начальное приближение сеточной функции
			w_prev[i][j] = 0;
		}
	}

	for (int j = 0; j <= N; ++j) {
		// сетка на прямоугольнике
		w2[j] = A2 + j * h2;
	}

	double **a = a_cmp(w1, w2, M, N, h1, h2), 
	       **b = b_cmp(w1, w2, M, N, h1, h2), 
		   **F = F_cmp(w1, w2, M, N, h1, h2);

	double tau, 
	       delta = 0.000001;

	if (M >= 100 && N >= 100) {
		delta = 1e-15;
	}

	// будем проверять не ||w^(k+1) - w^k|| < delta
	// а ||w^(k+1) - w^k||^2 < delta^2
	delta *= delta;

	// инициализация работы программы
	gettimeofday(&end, NULL);
	double elapsed_time_init = end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) / 1000000.0;

	// основная часть
	struct timeval start_loop, end_loop;
	gettimeofday(&start, NULL);

	// тело цикла do while целиком состоит из паралелльных циклов
	gettimeofday(&start_loop, NULL);
	int iter = 0;
	do {
		iter += 1;

		gettimeofday(&start_loop, NULL);

		// для экономии памяти не создаем переменные под промежуточные
		// матрицы при расчетах
		// Aw^k is written to Ar
		Aw_cmp(Ar, // result
			   w_prev, a, b, M, N, h1, h2);

		// r^k = Aw^k - F
		matr_diff(r, // result
			      Ar, F, M, N);

		gettimeofday(&end_loop, NULL);
		if (iter == 1) printf("first iter r=Aw-F %.10f\n", end_loop.tv_sec - start_loop.tv_sec + (end_loop.tv_usec - start_loop.tv_usec) / 1000000.0);

		// compute Ar^k
		gettimeofday(&start_loop, NULL);
		Aw_cmp(Ar, r, a, b, M, N, h1, h2);
		gettimeofday(&end_loop, NULL);
		if (iter == 1) printf("first iter Ar %.10f\n", end_loop.tv_sec - start_loop.tv_sec + (end_loop.tv_usec - start_loop.tv_usec) / 1000000.0);

		gettimeofday(&start_loop, NULL);
		tau = dot(Ar, r, M, N, h1, h2) / sqnorm(Ar, M, N, h1, h2);
		gettimeofday(&end_loop, NULL);
		if (iter == 1) printf("first iter (Ar,r) (Ar,Ar) %.10f\n", end_loop.tv_sec - start_loop.tv_sec + (end_loop.tv_usec - start_loop.tv_usec) / 1000000.0);
		

		gettimeofday(&start_loop, NULL);
		for (int i = 1; i <= M - 1; ++i) {
			for (int j = 1; j <= N - 1; ++j) {

				// w^(k+1) = w^k - tau*r^k
				// w^(k+1) - w^k = -tau*r^k

				// (w^(k+1) - w^k) is written to r
				r[i][j] *= -tau;

				// сразу пишем новую сетку w^(k+1)
				// в w_prev для следующей итерации
				w_prev[i][j] += r[i][j];
			}
		}
		gettimeofday(&end_loop, NULL);
		if (iter == 1) printf("first iter new grid %.10f\n", end_loop.tv_sec - start_loop.tv_sec + (end_loop.tv_usec - start_loop.tv_usec) / 1000000.0);


		gettimeofday(&start_loop, NULL);
		// ||w^(k+1) - w^k||^2 is written to tau
		tau = sqnorm(r, M, N, h1, h2);
		gettimeofday(&end_loop, NULL);
		if (iter == 1) printf("first iter diff norm %.10f\n", end_loop.tv_sec - start_loop.tv_sec + (end_loop.tv_usec - start_loop.tv_usec) / 1000000.0);

		printf("Current sqnorm %.35f\n", tau);
	}
	while (tau >= delta);

	gettimeofday(&end_loop, NULL);
	
	// основная часть
	gettimeofday(&end, NULL);
	double elapsed_time_main = end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) / 1000000.0;

	double loop_total_time = end_loop.tv_sec - start_loop.tv_sec + (end_loop.tv_usec - start_loop.tv_usec) / 1000000.0;

	// завершение работы
	gettimeofday(&start, NULL);

	free_a(a, M, N);
	free_b(b, M, N);
	free_F(F, M, N);

	if (!print_result) {
		free(w1);
		free(w2);
	}

	for (int i = 0; i <= M; ++i) {
		free(w_prev[i]);
		free(r[i]);
		free(Ar[i]);
	}

	if (!print_result) {
		free(w_prev);
	}

	free(r);
	free(Ar);

	// завершение работы
	gettimeofday(&end, NULL);
	double elapsed_time_end = end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) / 1000000.0;


	printf("Инициализация: %.6f секунд \nОсновная часть: %.6f секунд \nЗавершение: %.6f секунд \n Итого: %.6f \n",
		    elapsed_time_init,  elapsed_time_main,  elapsed_time_end,
		    elapsed_time_init + elapsed_time_main + elapsed_time_end);

	if (print_loop_cpy_exch_time) {
		printf("Циклы: %.6f\n",loop_total_time);
	}

	if (print_result) {
		printf("Значения точек сетки (первая строка и первый столбец) и сеточной функции:\n\n");
		print_net_function(w_prev, w1, w2, M, N);
		free(w1);
		free(w2);
		free(w_prev);
	}
	printf("%d\n", iter);
	return 0;
}