#include "cint.h"
#include "optimizer.h"

int CVHFrkbssll_prescreen(int *shls, CVHFOpt *opt, int *atm, int *bas, double *env);
int CVHFrkbllll_prescreen(int *shls, CVHFOpt *opt, int *atm, int *bas, double *env);

int CVHFrkb_a1n_noscreen(int *shls, CVHFOpt *opt,
                         int *atm, int *bas, double *env)
{
        int i = shls[0];
        int j = shls[1];
        int k = shls[2];
        int l = shls[3];
        int center_i = bas[i*BAS_SLOTS+ATOM_OF];
        int center_j = bas[j*BAS_SLOTS+ATOM_OF];
        int center_k = bas[k*BAS_SLOTS+ATOM_OF];
        int center_l = bas[l*BAS_SLOTS+ATOM_OF];
        if (center_i == center_j && center_i == center_k && center_i == center_l) {
                return 1;
        }
        return 0;
}

int CVHFrkbssll_a1n_prescreen(int *shls, CVHFOpt *opt,
                              int *atm, int *bas, double *env)
{
        int i = shls[0];
        int j = shls[1];
        int k = shls[2];
        int l = shls[3];
        int center_i = bas[i*BAS_SLOTS+ATOM_OF];
        int center_j = bas[j*BAS_SLOTS+ATOM_OF];
        int center_k = bas[k*BAS_SLOTS+ATOM_OF];
        int center_l = bas[l*BAS_SLOTS+ATOM_OF];
        if (center_i == center_j && center_i == center_k && center_i == center_l) {
                return CVHFrkbssll_prescreen(shls, opt, atm, bas, env);
        }
        return 0;
}

int CVHFrkbssss_a1n_prescreen(int *shls, CVHFOpt *opt,
                              int *atm, int *bas, double *env)
{
        int i = shls[0];
        int j = shls[1];
        int k = shls[2];
        int l = shls[3];
        int center_i = bas[i*BAS_SLOTS+ATOM_OF];
        int center_j = bas[j*BAS_SLOTS+ATOM_OF];
        int center_k = bas[k*BAS_SLOTS+ATOM_OF];
        int center_l = bas[l*BAS_SLOTS+ATOM_OF];
        if (center_i == center_j && center_i == center_k && center_i == center_l) {
                return CVHFrkbllll_prescreen(shls, opt, atm, bas, env);
        }
        return 0;
}

int CVHFrkb_a2n_noscreen(int *shls, CVHFOpt *opt,
                         int *atm, int *bas, double *env)
{
        int i = shls[0];
        int j = shls[1];
        int k = shls[2];
        int l = shls[3];
        int center_i = bas[i*BAS_SLOTS+ATOM_OF];
        int center_j = bas[j*BAS_SLOTS+ATOM_OF];
        int center_k = bas[k*BAS_SLOTS+ATOM_OF];
        int center_l = bas[l*BAS_SLOTS+ATOM_OF];
        if (center_i == center_j && center_k == center_l) {
                return 1;
        }
        return 0;
}

int CVHFrkbssll_a2n_prescreen(int *shls, CVHFOpt *opt,
                              int *atm, int *bas, double *env)
{
        int i = shls[0];
        int j = shls[1];
        int k = shls[2];
        int l = shls[3];
        int center_i = bas[i*BAS_SLOTS+ATOM_OF];
        int center_j = bas[j*BAS_SLOTS+ATOM_OF];
        int center_k = bas[k*BAS_SLOTS+ATOM_OF];
        int center_l = bas[l*BAS_SLOTS+ATOM_OF];
        if (center_i == center_j && center_k == center_l) {
                return CVHFrkbssll_prescreen(shls, opt, atm, bas, env);
        }
        return 0;
}

int CVHFrkbssss_a2n_prescreen(int *shls, CVHFOpt *opt,
                              int *atm, int *bas, double *env)
{
        int i = shls[0];
        int j = shls[1];
        int k = shls[2];
        int l = shls[3];
        int center_i = bas[i*BAS_SLOTS+ATOM_OF];
        int center_j = bas[j*BAS_SLOTS+ATOM_OF];
        int center_k = bas[k*BAS_SLOTS+ATOM_OF];
        int center_l = bas[l*BAS_SLOTS+ATOM_OF];
        if (center_i == center_j && center_k == center_l) {
                return CVHFrkbllll_prescreen(shls, opt, atm, bas, env);
        }
        return 0;
}
