/*
 * Author: Qiming Sun <osirpt.sun@gmail.com>
 */

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "config.h"
#include "cint.h"
#include "vhf/fblas.h"
#include "grid_ao_drv.h"

#define MIN(X,Y)        ((X)<(Y)?(X):(Y))
#define MAX(X,Y)        ((X)>(Y)?(X):(Y))

double exp_cephes(double x);
double CINTcommon_fac_sp(int l);

static int _len_cart[] = {
        1, 3, 6, 10, 15, 21, 28, 36
};

void GTOnabla1(double *fx1, double *fy1, double *fz1,
               double *fx0, double *fy0, double *fz0, int l, double a)
{
        int i;
        double a2 = -2 * a;
        fx1[0] = a2*fx0[1];
        fy1[0] = a2*fy0[1];
        fz1[0] = a2*fz0[1];
        for (i = 1; i <= l; i++) {
                fx1[i] = i*fx0[i-1] + a2*fx0[i+1];
                fy1[i] = i*fy0[i-1] + a2*fy0[i+1];
                fz1[i] = i*fz0[i-1] + a2*fz0[i+1];
        }
}

/*
 * r - R_O = (r-R_i) + ri, ri = (x,y,z) = R_i - R_O
 */
void GTOx1(double *fx1, double *fy1, double *fz1,
           double *fx0, double *fy0, double *fz0, int l, double *ri)
{
        int i;
        for (i = 0; i <= l; i++) {
                fx1[i] = ri[0] * fx0[i] + fx0[i+1];
                fy1[i] = ri[1] * fy0[i] + fy0[i+1];
                fz1[i] = ri[2] * fz0[i] + fz0[i+1];
        }
}

int GTOprim_exp(double *eprim, double *coord, double *alpha, double *coeff,
                int l, int nprim, int nctr, int blksize, double fac)
{
        int i, j;
        double arr, maxc;
        double logcoeff[nprim];
        double rr[blksize];
        double *gridx = coord;
        double *gridy = coord+blksize;
        double *gridz = coord+blksize*2;
        int not0 = 0;

        // the maximum value of the coefficients for each pGTO
        for (j = 0; j < nprim; j++) {
                maxc = 0;
                for (i = 0; i < nctr; i++) {
                        maxc = MAX(maxc, fabs(coeff[i*nprim+j]));
                }
                logcoeff[j] = log(maxc);
        }

        for (i = 0; i < blksize; i++) {
                rr[i] = gridx[i]*gridx[i] + gridy[i]*gridy[i] + gridz[i]*gridz[i];
        }

        for (j = 0; j < nprim; j++) {
                for (i = 0; i < blksize; i++) {
                        arr = alpha[j] * rr[i];
                        if (arr-logcoeff[j] < EXPCUTOFF) {
                                eprim[j*blksize+i] = exp_cephes(-arr) * fac;
                                not0 = 1;
                        } else {
                                eprim[j*blksize+i] = 0;
                        }
                }
        }
        return not0;
}


// grid2atm[atm_id,xyz,grid_id]
static void _fill_grid2atm(double *grid2atm, double *coord, int blksize, int ngrids,
                           int *atm, int natm, int *bas, int nbas, double *env)
{
        int atm_id, ig;
        double *r_atm;
        for (atm_id = 0; atm_id < natm; atm_id++) {
                r_atm = env + atm[PTR_COORD+atm_id*ATM_SLOTS];
                for (ig = 0; ig < blksize; ig++) {
                        grid2atm[0*blksize+ig] = coord[0*ngrids+ig] - r_atm[0];
                        grid2atm[1*blksize+ig] = coord[1*ngrids+ig] - r_atm[1];
                        grid2atm[2*blksize+ig] = coord[2*ngrids+ig] - r_atm[2];
                }
                grid2atm += 3*blksize;
        }
}


static void _trans(double *ao, double *aobuf, int nao, int blksize, int counts)
{
        int i, j, k;
        if (blksize == BLKSIZE) {
                for (k = 0; k < BLKSIZE; k+=16) {
                        for (i = 0; i < counts; i++) {
                                for (j = k; j < k+16; j++) {
                                        ao[j*nao+i] = aobuf[i*BLKSIZE+j];
                                }
                        }
                }
        } else if ((blksize % 16) == 0) {
                for (k = 0; k < blksize; k+=16) {
                        for (i = 0; i < counts; i++) {
                                for (j = k; j < k+16; j++) {
                                        ao[j*nao+i] = aobuf[i*blksize+j];
                                }
                        }
                }
        } else {
                for (i = 0; i < counts; i++) {
                        for (j = 0; j < blksize; j++) {
                                ao[j*nao+i] = aobuf[j];
                        }
                        aobuf += blksize;
                }
        }
}

static void _set0(double *out, int odim, int blksize, int counts)
{
        int i, j;
        for (i = 0; i < counts; i++) {
                for (j = 0; j < blksize; j++) {
                        out[i*odim+i] = 0;
                }
        }
}

static void _dcopy(double *out, double *in, int odim, int idim, int counts)
{
        int i, j;
        for (i = 0; i < counts; i++) {
                for (j = 0; j < idim; j++) {
                        out[i*odim+j] = in[i*idim+j];
                }
        }
}

void GTOeval_sph_iter(void (*feval)(),  int (*fexp)(),
                      int param[], int nao, int ngrids, int blksize,
                      int *shls_slice,
                      double *ao, double *coord, char *non0table,
                      int *atm, int natm, int *bas, int nbas, double *env)
{
        const int ncomp = param[TENSOR];
        const int sh0 = shls_slice[0];
        const int sh1 = shls_slice[1];
        const int atmstart = bas[sh0*BAS_SLOTS+ATOM_OF];
        const int atmend = bas[(sh1-1)*BAS_SLOTS+ATOM_OF]+1;
        const int atmcount = atmend - atmstart;
        int i, k, l, np, nc, atm_id, bas_id, deg;
        int ao_id = 0;
        double fac;
        double *p_exp, *pcoeff, *pcoord, *pcart, *ri, *pao;
        double eprim[NPRIMAX*blksize*2];
        double cart_gto[NCTR_CART*blksize * ncomp];
        double aobuf[(LMAX*2+2)*blksize];
        double grid2atm[atmcount*3*blksize]; // [atm_id,xyz,grid]

        _fill_grid2atm(grid2atm, coord, blksize, ngrids,
                       atm+atmstart*ATM_SLOTS, atmcount, bas, nbas, env);

        for (bas_id = sh0; bas_id < sh1; bas_id++) {
                np = bas[bas_id*BAS_SLOTS+NPRIM_OF];
                nc = bas[bas_id*BAS_SLOTS+NCTR_OF ];
                l  = bas[bas_id*BAS_SLOTS+ANG_OF  ];
                deg = l * 2 + 1;
                fac = CINTcommon_fac_sp(l);
                p_exp  = env + bas[bas_id*BAS_SLOTS+PTR_EXP];
                pcoeff = env + bas[bas_id*BAS_SLOTS+PTR_COEFF];
                atm_id = bas[bas_id*BAS_SLOTS+ATOM_OF];
                pcoord = grid2atm + (atm_id - atmstart) * 3*blksize;
                if (non0table[bas_id] &&
                    (*fexp)(eprim, pcoord, p_exp, pcoeff,
                            l, np, nc, blksize, fac)) {
                        ri = env + atm[PTR_COORD+atm_id*ATM_SLOTS];
                        if (l <= 1) { // s, p functions
                                (*feval)(ao+ao_id*ngrids, ri, eprim, pcoord, p_exp, pcoeff,
                                         l, np, nc, nao, ngrids, blksize);
                        } else {
                                (*feval)(cart_gto, ri, eprim, pcoord, p_exp, pcoeff,
                                         l, np, nc, nc*_len_cart[l], blksize, blksize);
                                pcart = cart_gto;
                                for (i = 0; i < ncomp; i++) {
                                        pao = ao + (i*nao+ao_id)*ngrids;
                                        for (k = 0; k < nc; k++) {
                                                CINTc2s_ket_sph(aobuf, blksize, pcart, l);
                                                _dcopy(pao+k*deg*ngrids, aobuf,
                                                       ngrids, blksize, deg);
                                                pcart += _len_cart[l]*blksize;
                                        }
                                }
                        }
                } else {
                        for (i = 0; i < ncomp; i++) {
                                _set0(ao+(i*nao+ao_id)*ngrids, ngrids, blksize, nc*deg);
                        }
                }
                ao_id += deg * nc;
        }
}

void GTOeval_cart_iter(void (*feval)(),  int (*fexp)(),
                       int param[], int nao, int ngrids, int blksize,
                       int *shls_slice,
                       double *ao, double *coord, char *non0table,
                       int *atm, int natm, int *bas, int nbas, double *env)
{
        const int ncomp = param[TENSOR];
        const int sh0 = shls_slice[0];
        const int sh1 = shls_slice[1];
        const int atmstart = bas[sh0*BAS_SLOTS+ATOM_OF];
        const int atmend = bas[(sh1-1)*BAS_SLOTS+ATOM_OF]+1;
        const int atmcount = atmend - atmstart;
        int i, k, l, np, nc, atm_id, bas_id, deg;
        int ao_id = 0;
        double fac;
        double *p_exp, *pcoeff, *pcoord, *pcart, *ri, *pao;
        double eprim[NPRIMAX*blksize*2];
        double grid2atm[atmcount*3*blksize]; // [atm_id,xyz,grid]

        _fill_grid2atm(grid2atm, coord, blksize, ngrids,
                       atm+atmstart*ATM_SLOTS, atmcount, bas, nbas, env);

        for (bas_id = sh0; bas_id < sh1; bas_id++) {
                np = bas[bas_id*BAS_SLOTS+NPRIM_OF];
                nc = bas[bas_id*BAS_SLOTS+NCTR_OF ];
                l  = bas[bas_id*BAS_SLOTS+ANG_OF  ];
                deg = _len_cart[l];
                fac = CINTcommon_fac_sp(l);
                p_exp  = env + bas[bas_id*BAS_SLOTS+PTR_EXP];
                pcoeff = env + bas[bas_id*BAS_SLOTS+PTR_COEFF];
                atm_id = bas[bas_id*BAS_SLOTS+ATOM_OF];
                pcoord = grid2atm + (atm_id - atmstart) * 3*blksize;
                if (non0table[bas_id] &&
                    (*fexp)(eprim, pcoord, p_exp, pcoeff,
                            l, np, nc, blksize, fac)) {
                        ri = env + atm[PTR_COORD+atm_id*ATM_SLOTS];
                        (*feval)(ao+ao_id*ngrids, ri, eprim, pcoord, p_exp, pcoeff,
                                 l, np, nc, nao, ngrids, blksize);
                } else {
                        for (i = 0; i < ncomp; i++) {
                                _set0(ao+(i*nao+ao_id)*ngrids, ngrids, blksize, nc*deg);
                        }
                }
                ao_id += deg * nc;
        }
}

static int _shloc_by_atom(int *shloc, int *shls_slice, int *ao_loc,
                          int *atm, int *bas)
{
        const int sh0 = shls_slice[0];
        const int sh1 = shls_slice[1];
        int ish, nshblk, lastatm;
        shloc[0] = sh0;
        nshblk = 1;
        lastatm = bas[ATM_SLOTS*sh0+ATOM_OF];
        for (ish = sh0; ish < sh1; ish++) {
                if (lastatm != bas[ATM_SLOTS*ish+ATOM_OF]) {
                        lastatm = bas[ATM_SLOTS*ish+ATOM_OF];
                        shloc[nshblk] = ish;
                        nshblk++;
                }
        }
        shloc[nshblk] = sh1;
        return nshblk;
}

/*
 * non0table[ngrids/blksize,natm] is the T/F table for ao values to
 * screen the ao evaluation for each shell
 */
void GTOeval_loop(void (*fiter)(), void (*feval)(), int (*fexp)(),
                  int param[], int *shls_slice, int *ao_loc, int ngrids,
                  double *ao, double *coord, char *non0table,
                  int *atm, int natm, int *bas, int nbas, double *env)
{
        int shloc[shls_slice[1]-shls_slice[0]+1];
        const int nshblk = _shloc_by_atom(shloc, shls_slice, ao_loc, atm, bas);
        const int nblk = (ngrids+BLKSIZE-1) / BLKSIZE;

#pragma omp parallel default(none) \
        shared(fiter, feval, fexp, param, ao_loc, shls_slice, ngrids, \
               ao, coord, non0table, atm, natm, bas, nbas, env, shloc)
{
        const int sh0 = shls_slice[0];
        const int sh1 = shls_slice[1];
        const int nao = ao_loc[sh1] - ao_loc[sh0];
        int ip, ib, k, iloc, ish;
#pragma omp for schedule(static, 8)
        for (k = 0; k < nblk*nshblk; k++) {
                iloc = k / nblk;
                ish = shloc[iloc];
                ib = k - iloc * nblk;
                ip = ib * BLKSIZE;
                (*fiter)(feval, fexp, param, nao, ngrids, MIN(ngrids-ip, BLKSIZE),
                         shloc+iloc, ao+ao_loc[ish]*ngrids+ip,
                         coord+ip, non0table+ib*nbas,
                         atm, natm, bas, nbas, env);
        }
}
}

void GTOeval_sph_drv(void (*feval)(), int (*fexp)(),
                     int param[], int *shls_slice, int *ao_loc, int ngrids,
                     double *ao, double *coord, char *non0table,
                     int *atm, int natm, int *bas, int nbas, double *env)
{
        GTOeval_loop(GTOeval_sph_iter, feval, fexp, param, shls_slice, ao_loc,
                     ngrids, ao, coord, non0table, atm, natm, bas, nbas, env);
}

void GTOeval_cart_drv(void (*feval)(), int (*fexp)(),
                      int param[], int *shls_slice, int *ao_loc, int ngrids,
                      double *ao, double *coord, char *non0table,
                      int *atm, int natm, int *bas, int nbas, double *env)
{
        GTOeval_loop(GTOeval_cart_iter, feval, fexp, param, shls_slice, ao_loc,
                     ngrids, ao, coord, non0table, atm, natm, bas, nbas, env);
}

