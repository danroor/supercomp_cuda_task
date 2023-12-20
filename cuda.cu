#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <mpi.h>
#include <cuda.h>
// #include <cuda_runtime.h>
// #include <curand.h>
// #include <curand_kernel.h>

// уравнение левой боковой стороны трапеции
__device__ double trap_left_y(double x) {
	return 3 * x + 9;
}
// x по y
__device__ double trap_left_x(double y) {
	return (y - 9) / 3;
}

// уравнение правой боковой стороны трапеции
__device__ double trap_right_y(double x) {
	return -3 * x + 9;
}
// x по y
__device__ double trap_right_x(double y) {
	return (9 - y) / 3;
}


__device__ int inside_trap(double x, double y) {

	return  (y >= 0) && (y <= 3) && 
	        (y <= trap_left_y(x)) && 
	        (y <= trap_right_y(x));

}

__global__ void abF_cmp(double *a, double *b, double *F,
	                    int M, int N, double h1, double h2,
	                    double A1, double A2,
	                    int local_imin, int local_jmin) {

	int linearIdx = threadIdx.x + blockIdx.x * blockDim.x;

	int i = linearIdx % (M + 1),
        j = linearIdx / (M + 1);

	if (i == 0 || j == 0 || linearIdx >= (M + 1) * (N + 1)) {
		return;
	}

	// сетка на прямоугольнике
	// переходим в глобальные координаты
	double x_i = A1 + (i + local_imin - 1) * h1,
	       y_j = A2 + (j + local_jmin - 1) * h2;

	double x_m, y_m, x_p, y_p, L, x_intersec, y_intersec;
	int P1_inside, P2_inside;

	double eps = h1 * h1;
	if (h2 > h1) {
		eps = h2 * h2;
	}

	// x_{i-1/2}
	x_m = x_i - h1 / 2;
	// x_{i+1/2}
	x_p = x_i + h1 / 2;
	// y_{j-1/2}
	y_m = y_j - h2 / 2;
	// y_{j+1/2}
	y_p = y_j + h2 / 2;

	// a_ij, i=1,M; j=1,N

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

	a[linearIdx] = (L + (h2 - L) / eps ) / h2;

	// b_ij, i=1,M; j=1,N

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

	b[linearIdx] = (L + (h1 - L) / eps ) / h1;

	// F_ij, i=1,M-1; j=1,N-1
	if (i == M || j == N) {
		return;
	}

	double y_x_m, y_x_p, x_y_p, x_y_m;

	// площадь каждого прямоугольника П_ij
	double S = h1 * h2;

	if (x_m * x_p <= 0) {
		// разные знаки координат или равенство одной из них нулю
		// при достаточно больших M, N (не менее 10) означают,
		// что прямоугольник П_ij в центре трапеции, далеко от 
		// ее боковых сторон, и полностью попадает внутрь трапеции
		F[linearIdx] = S;

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
		F[linearIdx] = S;
	} else if (y_x_m >= y_m) {
		if (y_x_p >= y_p) {
			// отрезаем треугольник
			F[linearIdx] = S - (y_p - y_x_m) * (x_y_p - x_m) / 2;
		} else if (y_x_m == y_m) {
			// отрезаем трапецию, справа остаётся треугольник
			F[linearIdx] = h1 * (y_x_p - y_m) / 2;
		} else {
			// справа остаётся трапеция
			F[linearIdx] = h1 * (y_x_m - y_m) + 
			          h1 * (y_x_p - y_x_m) / 2;
		}
	} else if (y_x_p > y_p) {
		// справа остаётся трапеция
		F[linearIdx] = h2 * (x_p - x_y_p) + 
		          h2 * (x_y_p - x_y_m) / 2;
	} else if (y_x_p > y_m) {
		// справа остаётся треугольник
		F[linearIdx] = (y_x_p - y_m) * (x_p - x_y_m) / 2;
	} else {
		// целиком снаружи
		F[linearIdx] = 0;
	}

	F[linearIdx] /= h1 * h2;

}

__device__ void Aw_cmp(
	// указатель на выделенную под результат память
	double *Aw,
	// сеточная функция, к которой применяется оператор
	double *w,
	double *a, double *b, int M, int N, double h1, double h2) {	

	int linearIdx = threadIdx.x + blockIdx.x * blockDim.x;
	int i = linearIdx % (M + 1),
        j = linearIdx / (M + 1);

	if (i == 0 || j == 0 || i == M || j == N || linearIdx >= (M + 1) * (N + 1)) {
		return;
	}

 	    // [i+1][j]
	int ip_j = j * (M + 1) + i + 1,
 	    // [i-1][j]
	    im_j = j * (M + 1) + i - 1,
 	    // [i][j+1]
	    i_jp = (j + 1) * (M + 1) + i,
 	    // [i][j-1]
	    i_jm = (j - 1) * (M + 1) + i;

	Aw[linearIdx] = -( a[ip_j] * (w[ip_j] - w[linearIdx]) - 
		          a[linearIdx] * (w[linearIdx]   - w[im_j]) ) / (h1*h1)

	           -( b[i_jp] * (w[i_jp] - w[linearIdx]) - 
		          b[linearIdx] *   (w[linearIdx]   - w[i_jm]) ) / (h2*h2);
}

__device__ void matr_diff(double *dst, double *m1, double *m2, int M, int N) {

	int linearIdx = threadIdx.x + blockIdx.x * blockDim.x;
	int i = linearIdx % (M + 1),
        j = linearIdx / (M + 1);

	if (i == 0 || j == 0 || i == M || j == N || linearIdx >= (M + 1) * (N + 1)) {
		return;
	}

	dst[linearIdx] = m1[linearIdx] - m2[linearIdx];
}

// r = Aw-F
__global__ void step1(double *r, double *Ar, double *w_prev, 
	                  double *a, double *b, double *F,
	                  int M, int N, double h1, double h2,
	                  int left, int right, int up, int down,
	                  double *lbuf_cuda, double *rbuf_cuda, double *ubuf_cuda, double *dbuf_cuda) {

	// Aw^k
	Aw_cmp(Ar, // result dst
		   w_prev, a, b, M, N, h1, h2);

	// r^k = Aw^k - F
	matr_diff(r, // result dst
		      Ar, F, M, N);

	int linearIdx = threadIdx.x + blockIdx.x * blockDim.x;
	int i = linearIdx % (M + 1),
        j = linearIdx / (M + 1);

    // выписываем внутренние границы r для соседних областей (i=1, j=1, i=M-1, j=N-1) 
    if (left != -1 && i == 1 && j >= 1 && j <= N - 1) {
    	lbuf_cuda[j] = r[linearIdx];
    } else if (right != -1 && i == M - 1 && j >= 1 && j <= N - 1) {
    	rbuf_cuda[j] = r[linearIdx];
    } else if (down != -1 && j == 1 && i >= 1 && i <= M - 1) {
    	dbuf_cuda[i] = r[linearIdx];
    } else if (up != -1 && j == N - 1 && i >= 1 && i <= M - 1) {
    	ubuf_cuda[i] = r[linearIdx];
    }

}

// Ar
__global__ void step2(double *r, double *Ar,
	                  double *a, double *b, 
	                  int M, int N, double h1, double h2) {

	// Ar
	Aw_cmp(Ar, // result dst
		   r, a, b, M, N, h1, h2);	


}

// w^(k+1), ||w^(k+1) - w^k||^2
__global__ void step3(double *r, double *w_prev, int M, int N, double tau,
					  int left, int right, int up, int down,
	                  double *lbuf_cuda, double *rbuf_cuda, double *ubuf_cuda, double *dbuf_cuda) {

		
	int linearIdx = threadIdx.x + blockIdx.x * blockDim.x;
	int i = linearIdx % (M + 1),
        j = linearIdx / (M + 1);
		
    if (i > 0 && i < M && j > 0 && j < N) {
		// w^(k+1) = w^k - tau*r^k
		// w^(k+1) - w^k = -tau*r^k
		// (w^(k+1) - w^k) is written to r
    	r[linearIdx] *= -tau;
    	w_prev[linearIdx] += r[linearIdx];

	    // выписываем внутренние границы w для соседних областей (i=1, j=1, i=M-1, j=N-1) 
	    if (left != -1 && i == 1 && j >= 1 && j <= N - 1) {
	    	lbuf_cuda[j] = w_prev[linearIdx];
	    } else if (right != -1 && i == M - 1 && j >= 1 && j <= N - 1) {
	    	rbuf_cuda[j] = w_prev[linearIdx];
	    } else if (down != -1 && j == 1 && i >= 1 && i <= M - 1) {
	    	dbuf_cuda[i] = w_prev[linearIdx];
	    } else if (up != -1 && j == N - 1 && i >= 1 && i <= M - 1) {
	    	ubuf_cuda[i] = w_prev[linearIdx];
	    }
    }
}

// скалярное произведение с редукцией по всем нитям
// редукция осуществляется по принципу бинарного дерева
// результат в a[0] (массив а изменяется)
// __global__ void dot_reduce(double *a,
//                            double *b,
//                             // шаг, с которым берутся ячейки для прибавления к ним соседей
//                             int step, 
//                             // длина массива
//                             int n) {
//     int idx = threadIdx.x + blockIdx.x * blockDim.x;

//     idx *= step;

//     int neighb_idx = idx + (step >> 1);

//     if (idx < n && neighb_idx < n) {
//         // домножение только при первом сложении
//         if (step == 2) {
//             a[idx] *= b[idx];    
//         }

//         // справа дополнительная проверка на случай нечётного размера массива
//         // и невхождения последнего элемента в сумму на первом вызове
//         if (step == 2 || neighb_idx == n - 1) {
//             a[neighb_idx] *= b[neighb_idx];
//         }

//         a[idx] += a[neighb_idx];
//     }
// }

__global__ void dot_reduce(double *a,
                           double *b,
                            // result
                           double *res,
                            // шаг, с которым берутся ячейки для прибавления к ним соседей
                            int step, 
                            // длина исходного массива
                            int n) {

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int r = (blockIdx.x + 1) * blockDim.x;

    r *= step;
    idx *= step;

    // считаем сумму по блокам
    // обрезаем неполный блок в конце
    if (n < r) {
        r = n;
    }

    while (idx + step < r) {

        int neighb_idx = idx + step;

        // idx % (2step) == 0
        if (idx % (step << 1) == 0 && idx < r && neighb_idx < r) {
            // домножение только при первом сложении
            if (step == 1) {
                res[idx] = a[idx] * b[idx];    
            }

            // справа дополнительная проверка на случай нечётного размера массива
            // и невхождения последнего элемента в сумму на первом вызове
            if (step == 1 || neighb_idx == r - 1) {
                res[neighb_idx] = a[neighb_idx] * b[neighb_idx];
            }

            res[idx] += res[neighb_idx];
        }

        step <<= 1; // *= 2


        __syncthreads();

    }
}

// суммирование с редукцией по всем нитям
// редукция осуществляется по принципу бинарного дерева
// результат в a[0] (массив а изменяется)
__global__ void sum_reduce(double *a,
                            // шаг, с которым берутся ячейки для прибавления к ним соседей
                            int step, 
                            // длина массива
                            int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    idx *= step;

    int neighb_idx = idx + (step >> 1);

    if (idx < n && neighb_idx < n) {
        a[idx] += a[neighb_idx];
    }
}


// переписываем внешние границы из буферных массивов в матрицу
__global__ void fill_borders(double *matr, double *lbuf_cuda, double *rbuf_cuda,  double *ubuf_cuda, double *dbuf_cuda,
	                         int M, int N, int left, int right, int up, int down) {

	int bufIdx = threadIdx.x + blockIdx.x * blockDim.x;

	if (bufIdx == 0) return;

	// bufIdx - номер элемента буферного массива
	// linearIdx  -линейный индекс граничного элемента в матрице
	int linearIdx;

	// проверка на случай N > M
	if (bufIdx <= M - 1) {
		// matr[bufIdx][0] -> get lin idx
		// linearIdx = 0 * (M + 1) + bufIdx = bufIdx
		if (down != -1) { matr[bufIdx] = dbuf_cuda[bufIdx]; }

		// matr[bufIdx][N] -> get lin idx
		linearIdx = N * (M + 1) + bufIdx;

		if (up   != -1) { matr[linearIdx] = ubuf_cuda[bufIdx]; }	
	}

	// проверка на случай M > N
	if (bufIdx <= N - 1) {
		// matr[0][bufIdx] -> get lin idx
		// linearIdx = bufIdx * (M + 1) + 0
		linearIdx = bufIdx * (M + 1);
		if (left  != -1) { matr[linearIdx] = lbuf_cuda[bufIdx]; }

		// matr[M][bufIdx] -> get lin idx
		linearIdx = bufIdx * (M + 1) + M;
		if (right != -1) { matr[linearIdx] = rbuf_cuda[bufIdx]; }
	}
}

void print_net_function(double *net, int M, int N,
	                    double A1, double A2, double h1, double h2) {

	printf("%15.10f ", -1.0);

	for (int i = 0; i <= M; ++i) {
		printf("%15.10f ", A1 + i * h1);
	}

	printf("\n");

	for (int j = 0; j <= N; ++j) {
		printf("%15.10f ", A2 + j * h2);
		for (int i = 0; i <= M; ++i) {
			printf("%15.10f ", net[j * (M + 1) + i]);
		}
		printf("\n");
	}	
}


double *buf;

// отсылаем соседям внутренние границы области
// получаем от соседей внешние границы
void exchange_borders_with_neighbors(double *lbuf, double *rbuf,  double *ubuf, double *dbuf,
									 int M, int N, int left, int right, int up, int down, int rank) {
	
	MPI_Status stat;
	int recv_count;

	if (right != -1) {

		for (int i = 1; i <= N - 1; ++i) {
			buf[i - 1] = rbuf[i];
		}

		MPI_Send(buf, N - 1, MPI_DOUBLE, right, rank,  MPI_COMM_WORLD);
		MPI_Recv(buf, N - 1, MPI_DOUBLE, right, right, MPI_COMM_WORLD, &stat);

		MPI_Get_count(&stat, MPI_DOUBLE, &recv_count);
		assert(recv_count == N - 1);

		for (int i = 1; i <= N - 1; ++i) {
			rbuf[i] = buf[i - 1];
		}

	}

	if (up != -1) {

		for (int i = 1; i <= M - 1; ++i) {
			buf[i - 1] = ubuf[i];
		}

		MPI_Send(buf, M - 1, MPI_DOUBLE, up, rank,  MPI_COMM_WORLD);
		MPI_Recv(buf, M - 1, MPI_DOUBLE, up, up,    MPI_COMM_WORLD, &stat);

		MPI_Get_count(&stat, MPI_DOUBLE, &recv_count);
		assert(recv_count == M - 1);

		for (int i = 1; i <= M - 1; ++i) {
			ubuf[i] = buf[i - 1];
		}

	}

	double x;

	if (left != -1) {
		MPI_Recv(buf, N - 1, MPI_DOUBLE, left, left, MPI_COMM_WORLD, &stat);

		MPI_Get_count(&stat, MPI_DOUBLE, &recv_count);
		assert(recv_count == N - 1);

		for (int i = 1; i <= N - 1; ++i) {
			x = lbuf[i];
			lbuf[i] = buf[i - 1];
			buf[i - 1] = x;
		}

		MPI_Send(buf, N - 1, MPI_DOUBLE, left, rank, MPI_COMM_WORLD);
	}

	if (down != -1) {
		MPI_Recv(buf, M - 1, MPI_DOUBLE, down, down,  MPI_COMM_WORLD, &stat);

		MPI_Get_count(&stat, MPI_DOUBLE, &recv_count);
		assert(recv_count == M - 1);

		for (int i = 1; i <= M - 1; ++i) {
			x = dbuf[i];
			dbuf[i] = buf[i - 1];
			buf[i - 1] = x;
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
		buf = (double *) calloc(N - 1, sizeof(double));
	} else {
		buf = (double *) calloc(M - 1, sizeof(double));
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

	// замеряем непосредственно вычисления по задаче Дирихле
	// поэтому ставим точку отсчета здесь

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

	// int M_global = M, N_global = N;

	M = imax + 1;
	N = jmax + 1;

	double *a, *b, *F;

	// добавим фиктивные строку и столбец с индексом 0 во избежание путаницы в индексации
	// размер a - (M+1)xN, так как присутствуют элементы a_M,j
	// размер b - Mx(N+1), так как присутствуют элементы b_i,N
	// размер F - MxN

	int n_dev, threads_per_block;
	cudaGetDeviceCount(&n_dev);
	cudaError_t err = cudaSetDevice(rank % n_dev);
	printf("Device setting error: %s\n", cudaGetErrorString(err));

	err = cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
	printf("Cache config setting error: %s\n", cudaGetErrorString(err));

	threads_per_block = 1024;

	int grid_num = (M + 1) * (N + 1);
	int grid_size = grid_num  * sizeof(double);

	cudaMalloc((void**)&a, grid_size);
	cudaMalloc((void**)&b, grid_size);
	cudaMalloc((void**)&F, grid_size);

	cudaMemset (a, 0, grid_size);
	cudaMemset (b, 0, grid_size);
	cudaMemset (F, 0, grid_size);

	abF_cmp<<< grid_num / threads_per_block + 1, threads_per_block>>>(a, b, F,
		 											  M, N, h1, h2,
		 											  A1, A2,
		 											  cur_proc_imin, cur_proc_jmin);

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

	double *w_prev, *r, *Ar, *dot_buf1, *dot_buf2,
			// границы подобластей для обмена
	       *lbuf, *rbuf, *ubuf, *dbuf,
	       *lbuf_cuda,*rbuf_cuda,  *ubuf_cuda, *dbuf_cuda;

	cudaMalloc((void**)&w_prev,   grid_size);
	cudaMalloc((void**)&r,        grid_size);
	cudaMalloc((void**)&Ar,       grid_size);
	cudaMalloc((void**)&dot_buf1, grid_size);
	cudaMalloc((void**)&dot_buf2, grid_size);

	cudaMemset (w_prev,   0, grid_size);
	cudaMemset (r,        0, grid_size);
	cudaMemset (Ar,       0, grid_size);
	cudaMemset (dot_buf1, 0, grid_size);
	cudaMemset (dot_buf2, 0, grid_size);

	int lrsize = N * sizeof(double),
	    udsize = M * sizeof(double);

	cudaMalloc((void**)&lbuf_cuda, lrsize);
	cudaMalloc((void**)&rbuf_cuda, lrsize);
	cudaMalloc((void**)&ubuf_cuda, udsize);
	cudaMalloc((void**)&dbuf_cuda, udsize);

	cudaMemset (lbuf_cuda, 0, lrsize);
	cudaMemset (rbuf_cuda, 0, lrsize);
	cudaMemset (ubuf_cuda, 0, udsize);
	cudaMemset (dbuf_cuda, 0, udsize);

	lbuf = (double *) calloc(N, sizeof(double));
	rbuf = (double *) calloc(N, sizeof(double));
	ubuf = (double *) calloc(M, sizeof(double));
	dbuf = (double *) calloc(M, sizeof(double));

	int maxMN = M;
	if (N > M) {
		maxMN = N;
	}

	// for dot_reduce
	int step, n_cpy, extra_block;

	cudaEvent_t event_start, event_stop;
	float elapsed_time;
	double loop_total_time = 0,
	       exch_total_time = 0,
	       cpy_total_time = 0,
	       elapsed_time_main = 0;
	
	int iter = 0;

	cudaStream_t stream1, stream2;
	cudaStreamCreate(&stream1);
	cudaStreamCreate(&stream2);

	// инициализация работы программы
	gettimeofday(&end, NULL);
	double elapsed_time_init = end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) / 1000000.0;
	// основная часть
	struct timeval start_mpi, end_mpi;
	gettimeofday(&start, NULL);

	do {
		if (print_loop_cpy_exch_time) {
			cudaEventCreate(&event_start);
			cudaEventCreate(&event_stop);
			cudaEventRecord(event_start, 0);
		}

		// r = Aw - F
		// на выходе из вызова в buf переменные записываются внутренние границы r (i=1, j=1, i=M-1, j=N-1) 
		step1<<< grid_num / threads_per_block + 1, threads_per_block >>>(r, Ar, w_prev, a, b, F, M, N, h1, h2,
			                                                             left, right, up, down, lbuf_cuda, rbuf_cuda,  ubuf_cuda, dbuf_cuda);

		if (print_loop_cpy_exch_time) {
			cudaEventRecord(event_stop, 0);
			cudaEventSynchronize(event_stop);
			cudaEventElapsedTime(&elapsed_time, event_start, event_stop);
			cudaEventDestroy(event_start);
			cudaEventDestroy(event_stop);

			loop_total_time += elapsed_time;
		}


		if (iter == 1) printf("first iter r=Aw-F %.10f\n", elapsed_time / 1000);


		if (print_loop_cpy_exch_time) {
			cudaEventCreate(&event_start);
			cudaEventCreate(&event_stop);
			cudaEventRecord(event_start, 0);
		}

		// обмен граничными значениями r с соседями
		if (left  != -1) cudaMemcpy(lbuf, lbuf_cuda, lrsize, cudaMemcpyDeviceToHost);
		if (right != -1) cudaMemcpy(rbuf, rbuf_cuda, lrsize, cudaMemcpyDeviceToHost);
		if (up    != -1) cudaMemcpy(ubuf, ubuf_cuda, udsize, cudaMemcpyDeviceToHost);
		if (down  != -1) cudaMemcpy(dbuf, dbuf_cuda, udsize, cudaMemcpyDeviceToHost);

		if (print_loop_cpy_exch_time) {
			cudaEventRecord(event_stop, 0);
			cudaEventSynchronize(event_stop);
			cudaEventElapsedTime(&elapsed_time, event_start, event_stop);
			cudaEventDestroy(event_start);
			cudaEventDestroy(event_stop);

			cpy_total_time += elapsed_time;
		}









		if (print_loop_cpy_exch_time) {
			gettimeofday(&start_mpi, NULL);
		}

		if (size > 1) {
			exchange_borders_with_neighbors(lbuf, rbuf,  ubuf, dbuf, M, N,
				                            left, right, up, down, rank);			
		}


		if (print_loop_cpy_exch_time) {
			gettimeofday(&end_mpi, NULL);
			elapsed_time = end_mpi.tv_sec - start_mpi.tv_sec + (end_mpi.tv_usec - start_mpi.tv_usec) / 1000000.0;
			exch_total_time += elapsed_time;
		}
	











		if (print_loop_cpy_exch_time) {
			cudaEventCreate(&event_start);
			cudaEventCreate(&event_stop);
			cudaEventRecord(event_start, 0);
		}

		if (left  != -1) cudaMemcpy(lbuf_cuda, lbuf, lrsize, cudaMemcpyHostToDevice);
		if (right != -1) cudaMemcpy(rbuf_cuda, rbuf, lrsize, cudaMemcpyHostToDevice);
		if (up    != -1) cudaMemcpy(ubuf_cuda, ubuf, udsize, cudaMemcpyHostToDevice);
		if (down  != -1) cudaMemcpy(dbuf_cuda, dbuf, udsize, cudaMemcpyHostToDevice);

		if (print_loop_cpy_exch_time) {
			cudaEventRecord(event_stop, 0);
			cudaEventSynchronize(event_stop);
			cudaEventElapsedTime(&elapsed_time, event_start, event_stop);
			cudaEventDestroy(event_start);
			cudaEventDestroy(event_stop);

			cpy_total_time += elapsed_time;
		}













		if (print_loop_cpy_exch_time) {
			cudaEventCreate(&event_start);
			cudaEventCreate(&event_stop);
			cudaEventRecord(event_start, 0);
		}

		// заполняем внешние границы r
		fill_borders <<< (maxMN - 1) / threads_per_block + 1, threads_per_block >>> (r, lbuf_cuda, rbuf_cuda,  ubuf_cuda, dbuf_cuda, M, N, left, right, up, down);

		// Вычисление Ar
		step2<<< grid_num / threads_per_block + 1, threads_per_block>>>(r, Ar, a, b, M, N, h1, h2);

		if (print_loop_cpy_exch_time) {
			cudaEventRecord(event_stop, 0);
			cudaEventSynchronize(event_stop);
			cudaEventElapsedTime(&elapsed_time, event_start, event_stop);
			cudaEventDestroy(event_start);
			cudaEventDestroy(event_stop);

			loop_total_time += elapsed_time;
		}


		if (iter == 1) printf("first iter fill r + Ar %.10f\n", elapsed_time / 1000);



		// (Ar,r), (Ar, Ar) без домножения на h1h2
	    step = 1;
	    n_cpy = grid_num;


		if (print_loop_cpy_exch_time) {
			cudaEventCreate(&event_start);
			cudaEventCreate(&event_stop);
			cudaEventRecord(event_start, 0);
		}

	    while (step < grid_num) {
	        dot_reduce <<< (n_cpy - 1) / threads_per_block + 1, threads_per_block, 0 , stream1>>> (Ar, r,  dot_buf1, step, grid_num);
	        dot_reduce <<< (n_cpy - 1) / threads_per_block + 1, threads_per_block, 0 , stream2>>> (Ar, Ar, dot_buf2, step, grid_num);
	        step *= threads_per_block;
	        extra_block = !!(n_cpy % threads_per_block);
	        n_cpy = n_cpy / threads_per_block + extra_block;
	    }

		if (print_loop_cpy_exch_time) {
			cudaEventRecord(event_stop, 0);
			cudaEventSynchronize(event_stop);
			cudaEventElapsedTime(&elapsed_time, event_start, event_stop);
			cudaEventDestroy(event_start);
			cudaEventDestroy(event_stop);

			loop_total_time += elapsed_time;
		}

		if (iter == 1) printf("first iter (Ar,r), (Ar,Ar) %.10f\n", elapsed_time / 1000);


		if (print_loop_cpy_exch_time) {
			cudaEventCreate(&event_start);
			cudaEventCreate(&event_stop);
			cudaEventRecord(event_start, 0);
		}

		cudaStreamSynchronize(stream1);
		cudaStreamSynchronize(stream2);

	    // результат в элементе [0]
	    cudaMemcpy(&local_dot,    dot_buf1, sizeof(double), cudaMemcpyDeviceToHost);
	    cudaMemcpy(&local_sqnorm, dot_buf2, sizeof(double), cudaMemcpyDeviceToHost);

		if (print_loop_cpy_exch_time) {
			cudaEventRecord(event_stop, 0);
			cudaEventSynchronize(event_stop);
			cudaEventElapsedTime(&elapsed_time, event_start, event_stop);
			cudaEventDestroy(event_start);
			cudaEventDestroy(event_stop);

			cpy_total_time += elapsed_time;
		}
	    









		if (print_loop_cpy_exch_time) {
			gettimeofday(&start_mpi, NULL);
		}

		if (size > 1) {
			MPI_Allreduce(&local_dot,    &global_dot,    1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
			MPI_Allreduce(&local_sqnorm, &global_sqnorm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);			
		} else {
			global_dot = local_dot;
			global_sqnorm =local_sqnorm;
		}


		if (print_loop_cpy_exch_time) {
			gettimeofday(&end_mpi, NULL);
			elapsed_time = end_mpi.tv_sec - start_mpi.tv_sec + (end_mpi.tv_usec - start_mpi.tv_usec) / 1000000.0;
			exch_total_time += elapsed_time;
		}












		tau =  global_dot / global_sqnorm;








		if (print_loop_cpy_exch_time) {
			cudaEventCreate(&event_start);
			cudaEventCreate(&event_stop);
			cudaEventRecord(event_start, 0);
		}

		// Вычисление новой сетки w^(k+1), запись разности w^(k+1) - w^k в r 
		// на выходе из вызова в buf переменные записываются внутренние границы w (i=1, j=1, i=M-1, j=N-1) 
		step3<<< grid_num / threads_per_block + 1, threads_per_block>>>(r, w_prev, M, N, tau,
																		left, right, up, down, lbuf_cuda, rbuf_cuda,  ubuf_cuda, dbuf_cuda);

		if (print_loop_cpy_exch_time) {
			cudaEventRecord(event_stop, 0);
			cudaEventSynchronize(event_stop);
			cudaEventElapsedTime(&elapsed_time, event_start, event_stop);
			cudaEventDestroy(event_start);
			cudaEventDestroy(event_stop);

			loop_total_time += elapsed_time;
		}


		
		if (iter == 1) printf("first iter new grid %.10f\n", elapsed_time / 1000);





		// ||w^(k+1) - w^k||^2 без домножения на h1h2
	    step = 1;
	    n_cpy = grid_num;



		if (print_loop_cpy_exch_time) {
			cudaEventCreate(&event_start);
			cudaEventCreate(&event_stop);
			cudaEventRecord(event_start, 0);
		}

	    while (step < grid_num) {
	        dot_reduce <<< (n_cpy - 1) / threads_per_block + 1, threads_per_block>>> (r, r, dot_buf1, step, grid_num);
	        step *= threads_per_block;
	        extra_block = !!(n_cpy % threads_per_block);
	        n_cpy = n_cpy / threads_per_block + extra_block;
	    }

		if (print_loop_cpy_exch_time) {
			cudaEventRecord(event_stop, 0);
			cudaEventSynchronize(event_stop);
			cudaEventElapsedTime(&elapsed_time, event_start, event_stop);
			cudaEventDestroy(event_start);
			cudaEventDestroy(event_stop);

			loop_total_time += elapsed_time;
		}





		if (iter == 1) printf("first iter diff norm %.10f\n", elapsed_time / 1000);




		if (print_loop_cpy_exch_time) {
			cudaEventCreate(&event_start);
			cudaEventCreate(&event_stop);
			cudaEventRecord(event_start, 0);
		}

	    cudaMemcpy(&local_sqnorm, dot_buf1, sizeof(double), cudaMemcpyDeviceToHost);

		if (print_loop_cpy_exch_time) {
			cudaEventRecord(event_stop, 0);
			cudaEventSynchronize(event_stop);
			cudaEventElapsedTime(&elapsed_time, event_start, event_stop);
			cudaEventDestroy(event_start);
			cudaEventDestroy(event_stop);

			cpy_total_time += elapsed_time;
		}
	    







		if (print_loop_cpy_exch_time) {
			gettimeofday(&start_mpi, NULL);
		}

		if (size > 1) {
			MPI_Allreduce(&local_sqnorm, &global_sqnorm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		} else {
			global_sqnorm = local_sqnorm;
		}

		if (print_loop_cpy_exch_time) {
			gettimeofday(&end_mpi, NULL);
			elapsed_time = end_mpi.tv_sec - start_mpi.tv_sec + (end_mpi.tv_usec - start_mpi.tv_usec) / 1000000.0;
			exch_total_time += elapsed_time;
		}
		










		global_sqnorm *= h1 * h2;













		if (print_loop_cpy_exch_time) {
			cudaEventCreate(&event_start);
			cudaEventCreate(&event_stop);
			cudaEventRecord(event_start, 0);
		}

		// Оператор Аw берет элементы невязки из соседней области,
		// обсчитываемой другим процессом
		// нужно обменяться граничными значениями сеточной функции с соседями

		cudaMemcpy(lbuf, lbuf_cuda, lrsize, cudaMemcpyDeviceToHost);
		cudaMemcpy(rbuf, rbuf_cuda, lrsize, cudaMemcpyDeviceToHost);
		cudaMemcpy(ubuf, ubuf_cuda, udsize, cudaMemcpyDeviceToHost);
		cudaMemcpy(dbuf, dbuf_cuda, udsize, cudaMemcpyDeviceToHost);

		if (print_loop_cpy_exch_time) {
			cudaEventRecord(event_stop, 0);
			cudaEventSynchronize(event_stop);
			cudaEventElapsedTime(&elapsed_time, event_start, event_stop);
			cudaEventDestroy(event_start);
			cudaEventDestroy(event_stop);

			cpy_total_time += elapsed_time;
		}









		if (print_loop_cpy_exch_time) {
			gettimeofday(&start_mpi, NULL);
		}

		if (size > 1) {
			exchange_borders_with_neighbors(lbuf, rbuf,  ubuf, dbuf, M, N,
				                            left, right, up, down, rank);
		}

		if (print_loop_cpy_exch_time) {
			gettimeofday(&end_mpi, NULL);
			elapsed_time = end_mpi.tv_sec - start_mpi.tv_sec + (end_mpi.tv_usec - start_mpi.tv_usec) / 1000000.0;
			exch_total_time += elapsed_time;
		}









		if (print_loop_cpy_exch_time) {
			cudaEventCreate(&event_start);
			cudaEventCreate(&event_stop);
			cudaEventRecord(event_start, 0);
		}

		// Оператор Аw берет элементы невязки из соседней области,
		// обсчитываемой другим процессом
		// нужно обменяться граничными значениями сеточной функции с соседями

		cudaMemcpy(lbuf_cuda, lbuf, lrsize, cudaMemcpyHostToDevice);
		cudaMemcpy(rbuf_cuda, rbuf, lrsize, cudaMemcpyHostToDevice);
		cudaMemcpy(ubuf_cuda, ubuf, udsize, cudaMemcpyHostToDevice);
		cudaMemcpy(dbuf_cuda, dbuf, udsize, cudaMemcpyHostToDevice);

		if (print_loop_cpy_exch_time) {
			cudaEventRecord(event_stop, 0);
			cudaEventSynchronize(event_stop);
			cudaEventElapsedTime(&elapsed_time, event_start, event_stop);
			cudaEventDestroy(event_start);
			cudaEventDestroy(event_stop);

			cpy_total_time += elapsed_time;
		}





		if (print_loop_cpy_exch_time) {
			cudaEventCreate(&event_start);
			cudaEventCreate(&event_stop);
			cudaEventRecord(event_start, 0);
		}

		// в buf внешние границы w от соседних областей (i=0, j=0, i=M, j=N)
		// w_prev заполняется этими значениями
		fill_borders <<< (maxMN - 1) / threads_per_block + 1, threads_per_block >>> (w_prev, lbuf_cuda, rbuf_cuda,  ubuf_cuda, dbuf_cuda, M, N, left, right, up, down);

		if (print_loop_cpy_exch_time) {
			cudaEventRecord(event_stop, 0);
			cudaEventSynchronize(event_stop);
			cudaEventElapsedTime(&elapsed_time, event_start, event_stop);
			cudaEventDestroy(event_start);
			cudaEventDestroy(event_stop);

			loop_total_time += elapsed_time;
		}

		if (iter == 1) printf("first iter fill w %.10f\n", elapsed_time / 1000);



		++iter;

		if (limit_iter != -1 && iter == limit_iter) {
			printf("Stopped after %d iterations\n", iter);
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

	cudaStreamDestroy(stream1);
	cudaStreamDestroy(stream2);

	cudaFree(a);
	cudaFree(b);
	cudaFree(F);

	// корректный замер завершения работы, если не выводится сетка
	if (!print_result) {
		cudaFree(w_prev);
	}

	cudaFree(r);
	cudaFree(Ar);
	cudaFree(dot_buf1);
	cudaFree(dot_buf2);

	cudaFree(lbuf_cuda);
	cudaFree(rbuf_cuda);
	cudaFree(ubuf_cuda);
	cudaFree(dbuf_cuda);

	free(lbuf);
	free(rbuf);
	free(ubuf);
	free(dbuf);

	free(buf);

	// завершение работы
	gettimeofday(&end, NULL);
	double elapsed_time_end = end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) / 1000000.0;

	// упорядоченный вывод результата всеми процессами от 0 до size-1
	MPI_Status stat;

	if (rank > 0) {
		MPI_Recv(&local_dot, 1, MPI_INT, rank - 1, rank - 1, MPI_COMM_WORLD, &stat);
	}

	printf("Процесс %d из %d успешно отработал с использованием GPU устройства %d из %d \nИнициализация: %.6f секунд \nОсновная часть: %.6f секунд \nЗавершение: %.6f секунд \n   Итого: %.6f \n",
		    rank, size, rank % n_dev, n_dev,
		    elapsed_time_init,  elapsed_time_main,  elapsed_time_end,
		    elapsed_time_init + elapsed_time_main + elapsed_time_end);

	// не учитываем накладные расходы на создание событий для замера времени выполнения команд cuda
	double real_time_main = loop_total_time / 1000 + cpy_total_time / 1000 + exch_total_time;

	if (print_loop_cpy_exch_time) {
		printf("Циклы: %.6f секунд, %.2f %% времени \nКопирования GPU: %.6f секунд, %.2f %% времени \nОбмены MPI: %.6f секунд, %.2f %% времени \n   Итого: %.6f",
		    loop_total_time / 1000, loop_total_time / 10 / real_time_main,
		    cpy_total_time / 1000, cpy_total_time / 10 / real_time_main,
		    exch_total_time, exch_total_time * 100 / real_time_main,
		    real_time_main);
	}

	// проверим корректность вычислений по левому верхнему домену
	if (print_result && left == -1 && up == -1) {
		printf("Значения точек сетки (первая строка и первый столбец) и сеточной функции:\n\n");
		double *w_host = (double *) calloc(grid_num, sizeof(double));
		cudaMemcpy(w_host, w_prev, grid_size, cudaMemcpyDeviceToHost);
		// в левой и нижней границах переходим в глобальные координаты
		print_net_function(w_host, M, N, A1 + (cur_proc_imin - 1) * h1, A2 + (cur_proc_jmin - 1) * h2, h1, h2);
	}

	printf("##############\n\n");

	if (print_result) {
		cudaFree(w_prev);
	}

	if (rank + 1 < size) {
		MPI_Send(&local_dot, 1, MPI_INT, rank + 1, rank,  MPI_COMM_WORLD);	
	}

	MPI_Finalize();
	return 0;
}
