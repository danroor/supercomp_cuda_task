#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

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
	#pragma omp parallel for
	for (int i = 1; i <= M; ++i) {
		free(a[i]);
	}
	free(a);
}

double **a_cmp(double *w1, double *w2, int M, int N, double h1, double h2) {
	// добавим фиктивные строку и столбец с индексом 0 во избежание путаницы в индексации
	double **a = calloc(M + 1, sizeof(double *));

	double eps = h1 * h1;
	if (h2 > h1) {
		eps = h2 * h2;
	}

	#pragma omp parallel for
	for (int i = 1; i <= M; ++i) {
		a[i] = calloc(N, sizeof(double));

		#pragma omp parallel for
		for (int j = 1; j <= N - 1; ++j) {

			double x_m, y_m, y_p, L, y_intersec;
			int P1_inside, P2_inside;

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
	#pragma omp parallel for
	for (int i = 1; i <= M - 1; ++i) {
		free(b[i]);
	}
	free(b);
}

double **b_cmp(double *w1, double *w2, int M, int N, double h1, double h2) {
	// добавим фиктивные строку и столбец с индексом 0 во избежание путаницы
	double **b = calloc(M, sizeof(double *));

	double eps = h1 * h1;
	if (h2 > h1) {
		eps = h2 * h2;
	}

	#pragma omp parallel for
	for (int i = 1; i <= M - 1; ++i) {
		b[i] = calloc(N + 1, sizeof(double));

		#pragma omp parallel for
		for (int j = 1; j <= N; ++j) {

			double y_m, x_m, x_p, L, x_intersec;
			int P1_inside, P2_inside;

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
	#pragma omp parallel for
	for (int i = 1; i <= M - 1; ++i) {
		free(F[i]);
	}
	free(F);
}

double **F_cmp(double *w1, double *w2, int M, int N, double h1, double h2) {

	double **F = calloc(M, sizeof(double *));

	// площадь каждого прямоугольника П_ij
	double S = h1 * h2;

	#pragma omp parallel for
	for (int i = 1; i <= M - 1; ++i) {
		F[i] = calloc(N, sizeof(double));
		#pragma omp parallel for
		for (int j = 1; j <= N - 1; ++j) {

			double x_m, y_m, x_p, y_p;
			double y_x_m, y_x_p, x_y_p, x_y_m;

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

	#pragma omp parallel for
	for (int i = 1; i <= M - 1; ++i) {
		#pragma omp parallel for
		for (int j = 1; j <= N - 1; ++j) {
			Aw[i][j] = -( a[i+1][j] * (w[i+1][j] - w[i][j]) - 
				          a[i][j] *   (w[i][j]   - w[i-1][j]) ) / (h1*h1)

			           -( b[i][j+1] * (w[i][j+1] - w[i][j]) - 
				          b[i][j] *   (w[i][j]   - w[i][j-1]) ) / (h2*h2);
		}
	}


}

void matr_diff(double **dst, double **m1, double **m2, int M, int N) {
	#pragma omp parallel for
	for (int i = 1; i <= M - 1; ++i) {
		#pragma omp parallel for
		for (int j = 1; j <= N - 1; ++j) {
			dst[i][j] = m1[i][j] - m2[i][j];
		}
	}
}

double dot(double **u, double **v, int M, int N, double h1, double h2) {
	double uv = 0;
	#pragma omp parallel for reduction (+:uv)
	for (int i = 1; i <= M - 1; ++i) {
		#pragma omp parallel for
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
	printf("%10.5f ", -1.0);

	for (int i = 0; i <= M; ++i) {
		printf("%10.5f ", w1[i]);
	}

	printf("\n");

	for (int j = 0; j <= N; ++j) {
		printf("%10.5f ", w2[j]);
		for (int i = 0; i <= M; ++i) {
			printf("%10.5f ", net[i][j]);
		}
		printf("\n");
	}	
}

int main(int argc, char *argv[])
{

	// поддержка вложенного паралеллизма
	omp_set_nested(1);
	printf("Программа выполняется на %d нитях\n\n", omp_get_max_threads());

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

	int print_result = 0;

	if (argc > 3 && strlen(argv[3]) == 1 && argv[3][0] == '1') {
		print_result = 1;
	}

	printf("M = %d, N = %d\n", M, N);

	// замеряем непосредственно вычисления по задаче Дирихле
	// поэтому ставим точку отсчета здесь
	
	struct timeval start, end;
	gettimeofday(&start, NULL);
	double start_omp = omp_get_wtime();

	// прямоугольник
	double A1 = -3, B1 = 3, A2 = 0, B2 = 3;

	double h1 = (B1 - A1) / M;
	double h2 = (B2 - A2) / N;

	double *w1 = calloc(M + 1, sizeof(double));
	double *w2 = calloc(N + 1, sizeof(double));

	double **w_prev = calloc(M + 1, sizeof(double *));
	double **r = calloc(M + 1, sizeof(double *));
	double **Ar = calloc(M + 1, sizeof(double *));

	#pragma omp parallel for
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

	// 0.3e-6 для сетки 160х160
	// 10^-6 для остальных сеток
	double tau, delta;

	if (M == 160 && N == 160) {
		delta = 0.3e-6;
	} else {
		delta = 1e-6;
	}

	// будем проверять не ||w^(k+1) - w^k|| < delta
	// а ||w^(k+1) - w^k||^2 < delta^2
	delta *= delta;

	do {
		// для экономии памяти не создаем переменные под промежуточные
		// матрицы при расчетах
		// Aw^k is written to Ar
		Aw_cmp(Ar, // result
			   w_prev, a, b, M, N, h1, h2);

		// r^k = Aw^k - F
		matr_diff(r, // result
			      Ar, F, M, N);

		// compute Ar^k
		Aw_cmp(Ar, r, a, b, M, N, h1, h2);

		tau = dot(Ar, r, M, N, h1, h2) / sqnorm(Ar, M, N, h1, h2);

		
		#pragma omp parallel for
		for (int i = 1; i <= M - 1; ++i) {
			#pragma omp parallel for
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

		// ||w^(k+1) - w^k||^2 is written to tau
		tau = sqnorm(r, M, N, h1, h2);
	}
	while (tau >= delta);

	gettimeofday(&end, NULL);
	double end_omp = omp_get_wtime();

	printf("Программа успешно выполнена за %.8f секунд (omp_get_wtime)\n",
		   end_omp - start_omp);

	printf("Программа успешно выполнена за %.8f секунд (gettimeofday)\n",
			end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) / 1000000.0);

	if (print_result) {
		printf("Значения точек сетки (первая строка и первый столбец) и сеточной функции:\n\n");
		print_net_function(w_prev, w1, w2, M, N);
	}

	free_a(a, M, N);
	free_b(b, M, N);
	free_F(F, M, N);

	free(w1);
	free(w2);

	#pragma omp parallel for
	for (int i = 0; i <= M; ++i) {
		free(w_prev[i]);
		free(r[i]);
		free(Ar[i]);
	}

	free(w_prev);
	free(r);
	free(Ar);

	return 0;
}