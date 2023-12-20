#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "mpi.h"

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



double *buf;

void exchange_borders_with_neighbors(double **grid,  int M, int N, int left, int right, int up, int down, int rank) {
	
	MPI_Status stat;
	int recv_count;

	if (right != -1) {

		for (int i = 1; i <= N - 1; ++i) {
			buf[i - 1] = grid[M - 1][i];
		}

		MPI_Send(buf, N - 1, MPI_DOUBLE, right, rank,  MPI_COMM_WORLD);
		MPI_Recv(buf, N - 1, MPI_DOUBLE, right, right, MPI_COMM_WORLD, &stat);

		MPI_Get_count(&stat, MPI_DOUBLE, &recv_count);
		assert(recv_count == N - 1);

		for (int i = 1; i <= N - 1; ++i) {
			grid[M][i] = buf[i - 1];
		}

	}

	if (up != -1) {

		for (int i = 1; i <= M - 1; ++i) {
			buf[i - 1] = grid[i][N - 1];
		}

		MPI_Send(buf, M - 1, MPI_DOUBLE, up, rank,  MPI_COMM_WORLD);
		MPI_Recv(buf, M - 1, MPI_DOUBLE, up, up,    MPI_COMM_WORLD, &stat);

		MPI_Get_count(&stat, MPI_DOUBLE, &recv_count);
		assert(recv_count == M - 1);

		for (int i = 1; i <= M - 1; ++i) {
			grid[i][N] = buf[i - 1];
		}

	}

	if (left != -1) {
		MPI_Recv(buf, N - 1, MPI_DOUBLE, left, left, MPI_COMM_WORLD, &stat);

		MPI_Get_count(&stat, MPI_DOUBLE, &recv_count);
		assert(recv_count == N - 1);

		for (int i = 1; i <= N - 1; ++i) {
			grid[0][i] = buf[i - 1];
			buf[i - 1] = grid[1][i];
		}

		MPI_Send(buf, N - 1, MPI_DOUBLE, left, rank, MPI_COMM_WORLD);
	}

	if (down != -1) {
		MPI_Recv(buf, M - 1, MPI_DOUBLE, down, down,  MPI_COMM_WORLD, &stat);

		MPI_Get_count(&stat, MPI_DOUBLE, &recv_count);
		assert(recv_count == M - 1);

		for (int i = 1; i <= M - 1; ++i) {
			grid[i][0] = buf[i - 1];
			buf[i - 1] = grid[i][1];
		}

		MPI_Send(buf, M - 1, MPI_DOUBLE, down, rank,  MPI_COMM_WORLD);
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

	int print_result = 0, print_loop_cpy_exch_time = 0, limit_iter = 0;

	if (argc > 3 && strlen(argv[3]) == 1 && argv[3][0] == '1') {
		print_result = 1;
	}

	if (argc > 4 && strlen(argv[4]) == 1 && argv[4][0] == '1') {
		print_loop_cpy_exch_time = 1;
	}

	if (argc > 5) {
		for (int i = 0; i < strlen(argv[5]); ++i) {
			// char digit to one-digit number
			digit = argv[5][i] - '0';
			if (digit < 0 || digit > 9) {
				printf("Неверный ограничитель итераций\n");
				return 0;		
			}

			limit_iter = limit_iter * 10 + digit;
		}	
	} else {
		limit_iter = -1;
	}

	printf("M = %d, N = %d\n", M, N);

	// для MPI обменов
	if (N > M) {
		buf = calloc(N - 1, sizeof(double));
	} else {
		buf = calloc(M - 1, sizeof(double));
	} 

	int rank,rank_cpy,
	    size,size_cpy,
	    // процесс отслеживает текущий рассматриваемый бит своего номера
	    rank_mask = 0,

	    // процесс запоминает номера процессов, обсчитывающих соседние домены
	    // (-1 - граница прямоугольника П)
	    left = -1,
	    right = -1,
	    up = -1,
	    down = -1,
	    
	    // процесс запоминает граничные индексы точек сетки, которые должен обсчитать
	    cur_proc_imin = 1,
	    cur_proc_imax = M - 1,
	    cur_proc_jmin = 1,
	    cur_proc_jmax = N - 1,
	    split_x = 1;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);


	rank_cpy = rank;
	size_cpy = size;
	

	if (N > M) {
		split_x = 0;
	}

	while (size_cpy > 1) {
		if (split_x) {
			// Разбиение вертикалью
			if (rank_cpy & 1) {
				// процесс берёт правую подобласть
				cur_proc_imin += (cur_proc_imax - cur_proc_imin) / 2 + 1;
				
				// новый сосед слева
				// младшие биты новому соседу процесс копирует от себя
				left = rank & rank_mask;
				
				// добавляем соседям по горизонтальной оси противоположный ведущий бит - ноль

				// добавляем соседям по вертикальной оси 
				// такой же ведущий бит - один
				// rank_mask = 111 - 3 бита уже просмотрели
				// rank_mask + 1 = 1000 - добавляем старший 4й бит
				if (up != -1) {
					up += rank_mask + 1;
				}		
				if (down != -1) {
					down += rank_mask + 1;
				}		

			} else {
				// процесс берёт левую подобласть
				cur_proc_imax = cur_proc_imin + (cur_proc_imax - cur_proc_imin) / 2;

				// новый сосед справа
				// младшие биты новому соседу процесс копирует от себя
				right = rank & rank_mask;
				
				// добавляем соседям по горизонтальной оси противоположный ведущий бит - один
				right += rank_mask + 1;
				
				if (left != -1) {
					left += rank_mask + 1;
				}

				// добавляем соседям по вертикальной оси 
				// такой же ведущий бит - ноль

			}

		} else {
			// Разбиение горизонталью
			if (rank_cpy & 1) {
				// процесс берёт верхнюю подобласть
				cur_proc_jmin += (cur_proc_jmax - cur_proc_jmin) / 2 + 1;

				// новый сосед снизу
				// младшие биты новому соседу процесс копирует от себя
				down = rank & rank_mask;
				 
				// добавляем соседям по вертикальной оси противоположный ведущий бит - ноль

				// добавляем соседям по горизонтальной оси 
				// такой же ведущий бит - один
				if (right != -1) {
					right += rank_mask + 1;
				}		
				if (left != -1) {
					left += rank_mask + 1;
				}	

			} else {
				// процесс берёт нижнюю подобласть
				cur_proc_jmax = cur_proc_jmin + (cur_proc_jmax - cur_proc_jmin) / 2;

				// новый сосед сверху
				// младшие биты новому соседу процесс копирует от себя
				up = rank & rank_mask;
				
				// добавляем соседям по вертикальной оси противоположный ведущий бит - один
				up += rank_mask + 1;
				
				if (down != -1) {
					down += rank_mask + 1;
				}

				// добавляем соседям по горизонтальной оси 
				// такой же ведущий бит - ноль

			}

		}

		rank_mask = (rank_mask << 1) + 1;

		size_cpy >>= 1;
		rank_cpy >>= 1;
		// Чередуем разбиения вертикалью и горизонталью
		split_x = !split_x;
	}

	// прямоугольник 
	double A1 = -3, B1 = 3, A2 = 0, B2 = 3;

	double h1 = (B1 - A1) / M;
	double h2 = (B2 - A2) / N;

	// [1..imax]x[1..jmax] - относительные (локальные) индексы точек сетки
	// [cur_proc_imin..cur_proc_imax]x[cur_proc_jmin..cur_proc_jmax] - глобальные индексы
	// 0, imax+1, jmax+1 - граница домена (точки из соседнего домена либо с границы прямоугольника П)
	// x_global = x_local + cur_proc_min - 1
	int imax = cur_proc_imax - cur_proc_imin + 1,
	    jmax = cur_proc_jmax - cur_proc_jmin + 1;

	// [0..imax+1]x[0..jmax+1]
	double *w1 = calloc(imax + 2, sizeof(double));
	double *w2 = calloc(jmax + 2, sizeof(double));

	double **w_prev = calloc(imax + 2, sizeof(double *));
	double **r = calloc(imax + 2, sizeof(double *));
	double **Ar = calloc(imax + 2, sizeof(double *));

	for (int i = 0; i <= imax + 1; ++i) {
		
		w_prev[i] = calloc(jmax + 2, sizeof(double));
		r[i] = calloc(jmax + 2, sizeof(double));
		Ar[i] = calloc(jmax + 2, sizeof(double));

		// сетка на прямоугольнике
		// переходим в глобальные координаты
		w1[i] = A1 + (i + cur_proc_imin - 1) * h1;
		
		for (int j = 0; j <= jmax + 1; ++j) {
			// начальное приближение сеточной функции
			w_prev[i][j] = 0;
		}
	}

	for (int j = 0; j <= jmax + 1; ++j) {
		// сетка на прямоугольнике
		w2[j] = A2 + (j + cur_proc_jmin - 1) * h2;
	}

	int M_global = M, N_global = N;

	M = imax + 1;
	N = jmax + 1;

	double tau,
	       local_dot, local_sqnorm,
		   global_dot, global_sqnorm,
	       delta = 0.000001;

	if (M >= 100 && N >= 100) {
		delta = 1e-15;
	}

	// будем проверять не ||w^(k+1) - w^k|| < delta
	// а ||w^(k+1) - w^k||^2 < delta^2
	delta *= delta;

	double elapsed_time,
	       loop_total_time = 0,
	       exch_total_time = 0,
	       elapsed_time_main = 0;

	// инициализация работы программы
	gettimeofday(&end, NULL);
	double elapsed_time_init = end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) / 1000000.0;
	// основная часть
	struct timeval start_mpi, end_mpi, start_loop, end_loop;
	gettimeofday(&start, NULL);

	double **a = a_cmp(w1, w2, M, N, h1, h2), 
	       **b = b_cmp(w1, w2, M, N, h1, h2), 
		   **F = F_cmp(w1, w2, M, N, h1, h2);

	int iter = 0;
	do {
		// для экономии памяти не создаем переменные под промежуточные
		// матрицы при расчетах
		// Aw^k is written to Ar

		if (print_loop_cpy_exch_time) {
			gettimeofday(&start_loop, NULL);
		}

		// на первой итерации везде нулевое начальное приближение
		// получать граничные значения w от соседей не нужно
		Aw_cmp(Ar, // result
			   w_prev, a, b, M, N, h1, h2);

		// r^k = Aw^k - F
		matr_diff(r, // result
			      Ar, F, M, N);

		if (print_loop_cpy_exch_time) {
			gettimeofday(&end_loop, NULL);
			elapsed_time = end_loop.tv_sec - start_loop.tv_sec + (end_loop.tv_usec - start_loop.tv_usec) / 1000000.0;
			loop_total_time += elapsed_time;
		}


		if (print_loop_cpy_exch_time) {
			gettimeofday(&start_mpi, NULL);
		}

		// Оператор А берет элементы невязки из соседней области,
		// обсчитываемой другим процессом
		// нужно обменяться граничными значениями с соседями
		exchange_borders_with_neighbors(r, M, N, left, right, up, down, rank);


		if (print_loop_cpy_exch_time) {
			gettimeofday(&end_mpi, NULL);
			elapsed_time = end_mpi.tv_sec - start_mpi.tv_sec + (end_mpi.tv_usec - start_mpi.tv_usec) / 1000000.0;
			exch_total_time += elapsed_time;
		}

		if (print_loop_cpy_exch_time) {
			gettimeofday(&start_loop, NULL);
		}

		// compute Ar^k
		Aw_cmp(Ar, r, a, b, M, N, h1, h2);

		local_dot = dot(Ar, r, M, N, h1, h2);
		local_sqnorm = sqnorm(Ar, M, N, h1, h2);
		
		if (print_loop_cpy_exch_time) {
			gettimeofday(&end_loop, NULL);
			elapsed_time = end_loop.tv_sec - start_loop.tv_sec + (end_loop.tv_usec - start_loop.tv_usec) / 1000000.0;
			loop_total_time += elapsed_time;
		}

		if (print_loop_cpy_exch_time) {
			gettimeofday(&start_mpi, NULL);
		}

		MPI_Allreduce(&local_dot, &global_dot, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		MPI_Allreduce(&local_sqnorm, &global_sqnorm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

		if (print_loop_cpy_exch_time) {
			gettimeofday(&end_mpi, NULL);
			elapsed_time = end_mpi.tv_sec - start_mpi.tv_sec + (end_mpi.tv_usec - start_mpi.tv_usec) / 1000000.0;
			exch_total_time += elapsed_time;
		}

		if (print_loop_cpy_exch_time) {
			gettimeofday(&start_loop, NULL);
		}

		tau =  global_dot / global_sqnorm;
		
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

		// ||w^(k+1) - w^k||^2
		local_sqnorm = sqnorm(r, M, N, h1, h2);

		if (print_loop_cpy_exch_time) {
			gettimeofday(&end_loop, NULL);
			elapsed_time = end_loop.tv_sec - start_loop.tv_sec + (end_loop.tv_usec - start_loop.tv_usec) / 1000000.0;
			loop_total_time += elapsed_time;
		}

		if (print_loop_cpy_exch_time) {
			gettimeofday(&start_mpi, NULL);
		}		

		MPI_Allreduce(&local_sqnorm, &global_sqnorm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

		// Оператор Аw берет элементы невязки из соседней области,
		// обсчитываемой другим процессом
		// нужно обменяться граничными значениями сеточной функции с соседями
		exchange_borders_with_neighbors(w_prev, M, N, left, right, up, down, rank);

		if (print_loop_cpy_exch_time) {
			gettimeofday(&end_mpi, NULL);
			elapsed_time = end_mpi.tv_sec - start_mpi.tv_sec + (end_mpi.tv_usec - start_mpi.tv_usec) / 1000000.0;
			exch_total_time += elapsed_time;
		}

		++iter;

		if (limit_iter != -1 && iter == limit_iter) {
			printf("Stopped after %d iterations\n", iter);
			break;
		}

		gettimeofday(&end, NULL);
		elapsed_time_main = end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) / 1000000.0;
		if (elapsed_time_main > 300) {
			printf("Stopped after %.6f seconds and %d iterations\n", elapsed_time_main, iter);
			break;
		}

		if (rank == 0) {
			printf("Current sqnorm %.35f\n", global_sqnorm);	
		}
		
	}
	while (global_sqnorm >= delta);

	// основная часть
	gettimeofday(&end, NULL);
	elapsed_time_main = end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) / 1000000.0;

	// завершение работы
	gettimeofday(&start, NULL);

	free_a(a, M, N);
	free_b(b, M, N);
	free_F(F, M, N);

	free(w1);
	free(w2);

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

	free(buf);

	// завершение работы
	gettimeofday(&end, NULL);
	double elapsed_time_end = end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) / 1000000.0;


	// упорядоченный вывод результата всеми процессами от 0 до size-1
	MPI_Status stat;

	if (rank > 0) {
		MPI_Recv(&local_dot, 1, MPI_INT, rank - 1, rank - 1, MPI_COMM_WORLD, &stat);
	}

	printf("Процесс %d из %d успешно отработал\nИнициализация: %.6f секунд \nОсновная часть: %.6f секунд \nЗавершение: %.6f секунд \n   Итого: %.6f \n",
		    rank, size,
		    elapsed_time_init,  elapsed_time_main,  elapsed_time_end,
		    elapsed_time_init + elapsed_time_main + elapsed_time_end);

	// не учитываем накладные расходы на создание событий для замера времени выполнения команд cuda
	double real_time_main = loop_total_time + exch_total_time;

	if (print_loop_cpy_exch_time) {
		printf("Циклы: %.6f секунд, %.2f %% времени \nОбмены MPI: %.6f секунд, %.2f %% времени \n   Итого: %.6f",
		    loop_total_time, loop_total_time * 100 / real_time_main,
		    exch_total_time, exch_total_time * 100 / real_time_main,
		    real_time_main);
	}

	// проверим корректность вычислений по левому верхнему домену
	if (print_result && left == -1 && up == -1) {
		printf("Значения точек сетки (первая строка и первый столбец) и сеточной функции:\n\n");
		print_net_function(w_prev, w1, w2, M, N);
	}

	printf("##############\n\n");


	if (print_result) {
		free(w_prev);
	}

	// printf("Процесс %d успешно отработал за %.8f секунд (gettimeofday)\n\n", rank,
	// 		end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) / 1000000.0);

	// // проверим корректность вычислений по левому верхнему домену
	// if (print_result && left == -1 && up == -1) {
	// 	printf("Значения точек сетки (первая строка и первый столбец) и сеточной функции:\n\n");
	// 	print_net_function(w_prev, w1, w2, M, N);
	// }

	MPI_Finalize();

	return 0;
}