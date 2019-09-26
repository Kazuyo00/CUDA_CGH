#define _USE_MATH_DEFINES
#define number 284
#define Wid 1920
#define Hgt 1080
#define tnum 1
#define device 0

#include <stdio.h>
#include <cmath>
#include <vector>
#include <sys/time.h>
#include <stdlib.h>
//#include <cuda.h>

#pragma pack(push,1)
typedef struct tagBITMAPFILEHEADER
{
	unsigned short bfType;
	int            bfSize;
	unsigned short bfReserved1;
	unsigned short bfReserved2;
	int            bf0ffBits;
}BITMAPFILEHEADER;

#pragma pack(pop)

typedef struct tagBITMAPINFOHEADER
{
	int             biSize;
	int			    biWid;
	int			    biHgt;
	unsigned short  biPlanes;
	unsigned short  biBitCount;
	int             biCompression;
	int             biSizeImage;
	int			    biXPelsPerMeter;
	int			    biYPelsPerMeter;
	int             biCirUsed;
	int             biCirImportant;
}BITMAPINFOHEADER;

typedef struct tagRGBQUAD
{
	unsigned char  rgbBlue;
	unsigned char  rgbGreen;
	unsigned char  rgbRed;
	unsigned char  rgbReserved;
}RGBQUAD;

typedef struct tagBITMAPINFO
{
	BITMAPINFOHEADER bmiHeader;
	RGBQUAD          bmiColors[1];
}BITMAPINFO;

/*時間計測*/
double gettimeofday_sec()
{
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return tv.tv_sec + tv.tv_usec * 1e-6;
}

/*カーネル関数*/
__global__ void holo_culculation(int *o_x, int *o_y, float *o_z, float *o_I){
	int i, j, k, adr;

	i = blockIdx.x*blockDim.x + threadIdx.x;
	j = blockIdx.y*blockDim.y + threadIdx.y;
	adr = i + j*Wid;

	float d_x, d_y, d_z, rr;
	float interval=10.5F;				//画素間隔
	float wave_len=0.633F;				//光波長
	float wave_num=2.0F*3.14159265F/wave_len;	//波数
	float kp=interval*wave_num;

	for(k=0; k<number; k++){
		d_x = ((float)j - o_x[k]) * ((float)j - o_x[k]);
		d_y = ((float)i - o_y[k]) * ((float)i - o_y[k]);
		d_z = o_z[k] * o_z[k];
		rr = sqrt(d_x + d_y + d_z);
		o_I[adr] = o_I[adr] + __cosf(kp*rr);
	}
}

unsigned char img[Hgt*Wid];
float I[Hgt*Wid];

int main()
{
	BITMAPFILEHEADER    BmpFileHeader;
	BITMAPINFOHEADER    BmpInfoHeader;
	RGBQUAD             RGBQuad[256];

	BmpFileHeader.bfType = 19778;
	BmpFileHeader.bfSize = 14 + 40 + 1024 + (256 * 256);
	BmpFileHeader.bfReserved1 = 0;
	BmpFileHeader.bfReserved2 = 0;
	BmpFileHeader.bf0ffBits = 14 + 40 + 1024;

	BmpInfoHeader.biSize = 40;
	BmpInfoHeader.biWid = Wid;
	BmpInfoHeader.biHgt = Hgt;
	BmpInfoHeader.biPlanes = 1;
	BmpInfoHeader.biBitCount = 8;
	BmpInfoHeader.biCompression = 0L;
	BmpInfoHeader.biSizeImage = 0L;
	BmpInfoHeader.biXPelsPerMeter = 0L;
	BmpInfoHeader.biYPelsPerMeter = 0L;
	BmpInfoHeader.biCirUsed = 0L;
	BmpInfoHeader.biCirImportant = 0L;

	int i, j, n;

	for (i = 0; i<256; i++){
		RGBQuad[i].rgbBlue = i;
		RGBQuad[i].rgbGreen = i;
		RGBQuad[i].rgbRed = i;
		RGBQuad[i].rgbReserved = 0;
	}

	cudaSetDevice(device);

	FILE *fp;

	fp = fopen("cube284.3d", "rb");
	fread(&n, sizeof(int), 1, fp);

	int x[number], y[number], x1, y1, z1;
	float z[number];

	for (i = 0; i<number; i++){
		fread(&x1, sizeof(int), 1, fp);
		fread(&y1, sizeof(int), 1, fp);
		fread(&z1, sizeof(int), 1, fp);

		x[i] = x1 * 40 + Hgt / 2;
		y[i] = y1 * 40 + Wid / 2;
		z[i] = (float)z1 * 40 + 50000.0F;
	}

	fclose(fp);

	double starttime, endtime, time_tmp;

	for (i = 0; i < Hgt; i++){
		for (j = 0; j < Wid; j++){
			I[i * Wid +j] = 0.0;
		}
	}

	starttime=gettimeofday_sec();

	int *o_x, *o_y;
	float *o_z, *o_I;

	dim3 block(32,8,1); //スレッド数(ブロック分割)
	dim3 grid(ceil(Wid/block.x),ceil(Hgt/block.y),1); //ブロック数(グリッド分割)

	cudaMalloc((void**)&o_x, number*sizeof(int));
	cudaMalloc((void**)&o_y, number*sizeof(int));
	cudaMalloc((void**)&o_z, number*sizeof(float));
	cudaMalloc((void**)&o_I, Wid*Hgt*sizeof(float));

	cudaMemcpy(o_x, x, number*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(o_y, y, number*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(o_z, z, number*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(o_I, I, Wid*Hgt*sizeof(float), cudaMemcpyHostToDevice);

	holo_culculation<<< grid, block >>>(o_x, o_y, o_z, o_I);

	cudaMemcpy(I, o_I, Wid*Hgt*sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(o_x);
	cudaFree(o_y);
	cudaFree(o_z);
	cudaFree(o_I);

	endtime=gettimeofday_sec();

	time_tmp = endtime-starttime;
	printf("%lf\n",time_tmp);

	time_tmp = 0.0;

	float max_tmp = 0.0, min_tmp = 0.0, mid_tmp = 0.0;

	max_tmp = I[0];
	min_tmp = I[0];

	for (i = 0; i < Hgt*Wid; i++){
		if (max_tmp <= I[i]){
			max_tmp = I[i];
//			printf("max i = %d\n",i);
		}

		else if (min_tmp > I[i]){
			min_tmp = I[i];
//			printf("min i = %d\n",i);
		}
	}

	mid_tmp = (max_tmp + min_tmp) * 0.5;

	printf("max = %lf\n", max_tmp);
	printf("min = %lf\n", min_tmp);
	printf("mid = %lf\n", mid_tmp);

	for (i = 0; i < Hgt*Wid; i++){
			img[i] = 0;
	}

	for (i = 0; i < Hgt*Wid; i++){

			if (I[i] <= mid_tmp){
				img[i] = 0;
			}

			else if (I[i] > mid_tmp){
				img[i] = 255;
			}
	}

	fp = fopen("CGH.bmp", "wb");

	fwrite(&BmpFileHeader, sizeof(BmpFileHeader), 1, fp);
	fwrite(&BmpInfoHeader, sizeof(BmpInfoHeader), 1, fp);
	fwrite(&RGBQuad[0], sizeof(RGBQuad[0]), 256, fp);
	fwrite(img, sizeof(unsigned char), Hgt * Wid, fp);

	fclose(fp);

	return 0;
}
