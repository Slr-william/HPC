#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>

#define NR 10
#define NC 10
#define MASTER 0
#define FROM_MASTER 1
#define FROM_WORKER 2

int main(int argc, char const *argv[]) {
  int nproc, numprocs, numworkers, piece, residuem, offset, mtype, rows, dest, source, i, j, k;
  double A[NR][NC],B[NR][NC],C[NR][NC];
  MPI_Status status;

  MPI_Init(&argc,&argv);
  MPI_Comm_rank(MPI_COMM_WORLD,&nproc);
  MPI_Comm_size(MPI_COMM_WORLD,&numprocs);

  if (numproc < 2 ) {
    printf("Need at least two MPI tasks. Quitting...\n");
    MPI_Abort(MPI_COMM_WORLD, rc);
    exit(1);
    }

    numworkers = numprocs-1;

  if (nproc == MASTER) {
    for (i=0; i<NR; i++)
       for (j=0; j<NC; j++){
          A[i][j]= 1;//i+j;
          B[i][j]= 2;//i*j;
        }
    piece = NR/numworkers;
    residue = NR%numworkers;
    offset = 0;
    mtype = FROM_MASTER;

    for (dest=1; dest<=numworkers; dest++){

       int rows = (dest <= residue) ? piece+1 : piece;
       printf("Sending %d rows to task %d offset=%d\n",rows,dest,offset);
       MPI_Send(&offset, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
       MPI_Send(&rows, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
       MPI_Send(&a[offset][0], rows*NC, MPI_DOUBLE, dest, mtype,
                 MPI_COMM_WORLD);
       MPI_Send(&b, NR*NC, MPI_DOUBLE, dest, mtype, MPI_COMM_WORLD);
       offset = offset + rows;
    }

    mtype = FROM_WORKER;
    for (i=1; i<=numworkers; i++)
    {
       source = i;
       MPI_Recv(&offset, 1, MPI_INT, source, mtype, MPI_COMM_WORLD, &status);
       MPI_Recv(&rows, 1, MPI_INT, source, mtype, MPI_COMM_WORLD, &status);
       MPI_Recv(&c[offset][0], rows*NC, MPI_DOUBLE, source, mtype,MPI_COMM_WORLD, &status);
    }
    printf("******************************************************\n");
    printf("Result Matrix:\n");
    for (i=0; i<NR; i++)
    {
       printf("\n");
       for (j=0; j<NC; j++)
          printf("%6.2f   ", c[i][j]);
    }
    printf("\n******************************************************\n");
    printf ("Done.\n");
  }

  if (nproc > MASTER) {
    mtype = FROM_MASTER;
    MPI_Recv(&offset, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
    MPI_Recv(&rows, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
    MPI_Recv(&a, rows*NC, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD, &status);
    MPI_Recv(&b, NCA*NC, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD, &status);

    for (k=0; k<NC; k++)
       for (i=0; i<rows; i++)
       {
          c[i][k] = 0.0;
          for (j=0; j<NC; j++)
             c[i][k] = c[i][k] + a[i][j] * b[j][k];
       }
     mtype = FROM_WORKER;
     MPI_Send(&offset, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD);
     MPI_Send(&rows, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD);
     MPI_Send(&c, rows*NCB, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
  }
  MPI_Finalize();
}
