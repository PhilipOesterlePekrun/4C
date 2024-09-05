/*----------------------------------------------------------------------------*/
/*! \file
\brief This file contains functions for the NURBS Kirchhoff-Love shell which are generated
with AceGen.

\level 1
*/
/*----------------------------------------------------------------------*/


#include "4C_fem_general_utils_nurbs_shapefunctions.hpp"
#include "4C_shell_kl_nurbs.hpp"

FOUR_C_NAMESPACE_OPEN



void Discret::ELEMENTS::KirchhoffLoveShellNurbs::evaluate_residuum_auto_generated(
    const double young, const double poisson, const double thickness,
    const Core::FE::IntegrationPoints1D& intpointsXi,
    const Core::FE::IntegrationPoints1D& intpointsEta,
    const std::vector<Core::LinAlg::SerialDenseVector>& knots,
    const Core::LinAlg::Matrix<9, 1>& weights, const Core::LinAlg::Matrix<9, 3>& X,
    const Core::LinAlg::Matrix<27, 1>& q, Core::LinAlg::SerialDenseVector& res)
{
  std::vector<double> v(1081);
  int i89, i90, i91, i92, i420;
  v[2] = poisson;
  v[642] = young / (1e0 - (v[2] * v[2]));
  v[275] = (1e0 - v[2]) / 2e0;
  v[3] = thickness;
  v[280] = (std::pow(v[3], 3) * v[642]) / 12e0;
  v[282] = v[275] * v[280];
  v[281] = v[2] * v[280];
  v[276] = v[3] * v[642];
  v[278] = v[275] * v[276];
  v[277] = v[2] * v[276];
  Core::LinAlg::Matrix<2, 1, double> uv;
  Core::LinAlg::Matrix<9, 1, double> N;
  Core::LinAlg::Matrix<2, 9, double> dN;
  Core::LinAlg::Matrix<3, 9, double> ddN;
  v[8] = X(0, 0);
  v[9] = X(0, 1);
  v[10] = X(0, 2);
  v[11] = X(1, 0);
  v[12] = X(1, 1);
  v[13] = X(1, 2);
  v[14] = X(2, 0);
  v[15] = X(2, 1);
  v[16] = X(2, 2);
  v[17] = X(3, 0);
  v[18] = X(3, 1);
  v[19] = X(3, 2);
  v[20] = X(4, 0);
  v[21] = X(4, 1);
  v[22] = X(4, 2);
  v[23] = X(5, 0);
  v[24] = X(5, 1);
  v[25] = X(5, 2);
  v[26] = X(6, 0);
  v[27] = X(6, 1);
  v[28] = X(6, 2);
  v[29] = X(7, 0);
  v[30] = X(7, 1);
  v[31] = X(7, 2);
  v[32] = X(8, 0);
  v[33] = X(8, 1);
  v[34] = X(8, 2);
  v[35] = q(0);
  v[36] = q(1);
  v[37] = q(2);
  v[38] = q(3);
  v[39] = q(4);
  v[40] = q(5);
  v[41] = q(6);
  v[42] = q(7);
  v[43] = q(8);
  v[44] = q(9);
  v[45] = q(10);
  v[46] = q(11);
  v[47] = q(12);
  v[48] = q(13);
  v[49] = q(14);
  v[50] = q(15);
  v[51] = q(16);
  v[52] = q(17);
  v[53] = q(18);
  v[54] = q(19);
  v[55] = q(20);
  v[56] = q(21);
  v[57] = q(22);
  v[58] = q(23);
  v[59] = q(24);
  v[60] = q(25);
  v[61] = q(26);
  i89 = intpointsXi.nquad;
  i90 = intpointsEta.nquad;
  for (i91 = 1; i91 <= i89; i91++)
  {
    v[644] = intpointsXi.qwgt[i91 - 1];
    v[93] = intpointsXi.qxg[i91 - 1][0];
    for (i92 = 1; i92 <= i90; i92++)
    {
      v[95] = intpointsEta.qwgt[i92 - 1] * v[644];
      uv(0) = v[93];
      uv(1) = intpointsEta.qxg[i92 - 1][0];
      Core::FE::Nurbs::nurbs_get_2d_funct_deriv_deriv2(
          N, dN, ddN, uv, knots, weights, Core::FE::CellType::nurbs9);
      v[99] = ddN(0, 0);
      v[100] = ddN(0, 1);
      v[101] = ddN(0, 2);
      v[102] = ddN(0, 3);
      v[103] = ddN(0, 4);
      v[104] = ddN(0, 5);
      v[105] = ddN(0, 6);
      v[106] = ddN(0, 7);
      v[107] = ddN(0, 8);
      v[216] = v[100] * v[11] + v[101] * v[14] + v[102] * v[17] + v[103] * v[20] + v[104] * v[23] +
               v[105] * v[26] + v[106] * v[29] + v[107] * v[32] + v[8] * v[99];
      v[231] = v[216] + v[100] * v[38] + v[101] * v[41] + v[102] * v[44] + v[103] * v[47] +
               v[104] * v[50] + v[105] * v[53] + v[106] * v[56] + v[107] * v[59] + v[35] * v[99];
      v[215] = v[100] * v[12] + v[101] * v[15] + v[102] * v[18] + v[103] * v[21] + v[104] * v[24] +
               v[105] * v[27] + v[106] * v[30] + v[107] * v[33] + v[9] * v[99];
      v[230] = v[215] + v[100] * v[39] + v[101] * v[42] + v[102] * v[45] + v[103] * v[48] +
               v[104] * v[51] + v[105] * v[54] + v[106] * v[57] + v[107] * v[60] + v[36] * v[99];
      v[214] = v[100] * v[13] + v[101] * v[16] + v[102] * v[19] + v[103] * v[22] + v[104] * v[25] +
               v[105] * v[28] + v[106] * v[31] + v[107] * v[34] + v[10] * v[99];
      v[229] = v[214] + v[100] * v[40] + v[101] * v[43] + v[102] * v[46] + v[103] * v[49] +
               v[104] * v[52] + v[105] * v[55] + v[106] * v[58] + v[107] * v[61] + v[37] * v[99];
      v[108] = ddN(1, 0);
      v[109] = ddN(1, 1);
      v[110] = ddN(1, 2);
      v[111] = ddN(1, 3);
      v[112] = ddN(1, 4);
      v[113] = ddN(1, 5);
      v[114] = ddN(1, 6);
      v[115] = ddN(1, 7);
      v[116] = ddN(1, 8);
      v[222] = v[109] * v[11] + v[110] * v[14] + v[111] * v[17] + v[112] * v[20] + v[113] * v[23] +
               v[114] * v[26] + v[115] * v[29] + v[116] * v[32] + v[108] * v[8];
      v[243] = v[222] + v[108] * v[35] + v[109] * v[38] + v[110] * v[41] + v[111] * v[44] +
               v[112] * v[47] + v[113] * v[50] + v[114] * v[53] + v[115] * v[56] + v[116] * v[59];
      v[221] = v[109] * v[12] + v[110] * v[15] + v[111] * v[18] + v[112] * v[21] + v[113] * v[24] +
               v[114] * v[27] + v[115] * v[30] + v[116] * v[33] + v[108] * v[9];
      v[242] = v[221] + v[108] * v[36] + v[109] * v[39] + v[110] * v[42] + v[111] * v[45] +
               v[112] * v[48] + v[113] * v[51] + v[114] * v[54] + v[115] * v[57] + v[116] * v[60];
      v[220] = v[10] * v[108] + v[109] * v[13] + v[110] * v[16] + v[111] * v[19] + v[112] * v[22] +
               v[113] * v[25] + v[114] * v[28] + v[115] * v[31] + v[116] * v[34];
      v[241] = v[220] + v[108] * v[37] + v[109] * v[40] + v[110] * v[43] + v[111] * v[46] +
               v[112] * v[49] + v[113] * v[52] + v[114] * v[55] + v[115] * v[58] + v[116] * v[61];
      v[117] = ddN(2, 0);
      v[118] = ddN(2, 1);
      v[119] = ddN(2, 2);
      v[120] = ddN(2, 3);
      v[121] = ddN(2, 4);
      v[122] = ddN(2, 5);
      v[123] = ddN(2, 6);
      v[124] = ddN(2, 7);
      v[125] = ddN(2, 8);
      v[219] = v[11] * v[118] + v[119] * v[14] + v[120] * v[17] + v[121] * v[20] + v[122] * v[23] +
               v[123] * v[26] + v[124] * v[29] + v[125] * v[32] + v[117] * v[8];
      v[237] = v[219] + v[117] * v[35] + v[118] * v[38] + v[119] * v[41] + v[120] * v[44] +
               v[121] * v[47] + v[122] * v[50] + v[123] * v[53] + v[124] * v[56] + v[125] * v[59];
      v[218] = v[118] * v[12] + v[119] * v[15] + v[120] * v[18] + v[121] * v[21] + v[122] * v[24] +
               v[123] * v[27] + v[124] * v[30] + v[125] * v[33] + v[117] * v[9];
      v[236] = v[218] + v[117] * v[36] + v[118] * v[39] + v[119] * v[42] + v[120] * v[45] +
               v[121] * v[48] + v[122] * v[51] + v[123] * v[54] + v[124] * v[57] + v[125] * v[60];
      v[217] = v[10] * v[117] + v[118] * v[13] + v[119] * v[16] + v[120] * v[19] + v[121] * v[22] +
               v[122] * v[25] + v[123] * v[28] + v[124] * v[31] + v[125] * v[34];
      v[235] = v[217] + v[117] * v[37] + v[118] * v[40] + v[119] * v[43] + v[120] * v[46] +
               v[121] * v[49] + v[122] * v[52] + v[123] * v[55] + v[124] * v[58] + v[125] * v[61];
      v[126] = dN(0, 0);
      v[127] = dN(0, 1);
      v[128] = dN(0, 2);
      v[129] = dN(0, 3);
      v[130] = dN(0, 4);
      v[131] = dN(0, 5);
      v[132] = dN(0, 6);
      v[133] = dN(0, 7);
      v[134] = dN(0, 8);
      v[169] = v[10] * v[126] + v[127] * v[13] + v[128] * v[16] + v[129] * v[19] + v[130] * v[22] +
               v[131] * v[25] + v[132] * v[28] + v[133] * v[31] + v[134] * v[34];
      v[181] = v[169] + v[126] * v[37] + v[127] * v[40] + v[128] * v[43] + v[129] * v[46] +
               v[130] * v[49] + v[131] * v[52] + v[132] * v[55] + v[133] * v[58] + v[134] * v[61];
      v[167] = v[12] * v[127] + v[128] * v[15] + v[129] * v[18] + v[130] * v[21] + v[131] * v[24] +
               v[132] * v[27] + v[133] * v[30] + v[134] * v[33] + v[126] * v[9];
      v[179] = v[167] + v[126] * v[36] + v[127] * v[39] + v[128] * v[42] + v[129] * v[45] +
               v[130] * v[48] + v[131] * v[51] + v[132] * v[54] + v[133] * v[57] + v[134] * v[60];
      v[165] = v[11] * v[127] + v[128] * v[14] + v[129] * v[17] + v[130] * v[20] + v[131] * v[23] +
               v[132] * v[26] + v[133] * v[29] + v[134] * v[32] + v[126] * v[8];
      v[177] = v[165] + v[126] * v[35] + v[127] * v[38] + v[128] * v[41] + v[129] * v[44] +
               v[130] * v[47] + v[131] * v[50] + v[132] * v[53] + v[133] * v[56] + v[134] * v[59];
      v[135] = dN(1, 0);
      v[136] = dN(1, 1);
      v[137] = dN(1, 2);
      v[138] = dN(1, 3);
      v[139] = dN(1, 4);
      v[140] = dN(1, 5);
      v[141] = dN(1, 6);
      v[142] = dN(1, 7);
      v[143] = dN(1, 8);
      v[170] = v[10] * v[135] + v[13] * v[136] + v[137] * v[16] + v[138] * v[19] + v[139] * v[22] +
               v[140] * v[25] + v[141] * v[28] + v[142] * v[31] + v[143] * v[34];
      v[182] = v[170] + v[135] * v[37] + v[136] * v[40] + v[137] * v[43] + v[138] * v[46] +
               v[139] * v[49] + v[140] * v[52] + v[141] * v[55] + v[142] * v[58] + v[143] * v[61];
      v[168] = v[12] * v[136] + v[137] * v[15] + v[138] * v[18] + v[139] * v[21] + v[140] * v[24] +
               v[141] * v[27] + v[142] * v[30] + v[143] * v[33] + v[135] * v[9];
      v[180] = v[168] + v[135] * v[36] + v[136] * v[39] + v[137] * v[42] + v[138] * v[45] +
               v[139] * v[48] + v[140] * v[51] + v[141] * v[54] + v[142] * v[57] + v[143] * v[60];
      v[166] = v[11] * v[136] + v[137] * v[14] + v[138] * v[17] + v[139] * v[20] + v[140] * v[23] +
               v[141] * v[26] + v[142] * v[29] + v[143] * v[32] + v[135] * v[8];
      v[178] = v[166] + v[135] * v[35] + v[136] * v[38] + v[137] * v[41] + v[138] * v[44] +
               v[139] * v[47] + v[140] * v[50] + v[141] * v[53] + v[142] * v[56] + v[143] * v[59];
      v[209] = -(v[178] * v[179]) + v[177] * v[180];
      v[207] = v[178] * v[181] - v[177] * v[182];
      v[183] = (v[165] * v[165]) + (v[167] * v[167]) + (v[169] * v[169]);
      v[251] = 1e0 / sqrt(v[183]);
      v[643] = 2e0 * v[251];
      v[266] = (v[251] * v[251]);
      v[184] = v[165] * v[166] + v[167] * v[168] + v[169] * v[170];
      v[185] = (v[166] * v[166]) + (v[168] * v[168]) + (v[170] * v[170]);
      v[199] = -(v[184] * v[184]) + v[183] * v[185];
      v[418] = sqrt(v[199]);
      v[187] = v[185] / v[199];
      v[188] = -(v[184] / v[199]);
      v[189] = v[183] / v[199];
      v[200] = (-(v[168] * v[169]) + v[167] * v[170]) / v[418];
      v[202] = (v[166] * v[169] - v[165] * v[170]) / v[418];
      v[203] = (-(v[166] * v[167]) + v[165] * v[168]) / v[418];
      v[204] = -(v[180] * v[181]) + v[179] * v[182];
      v[381] = (v[204] * v[204]) + (v[207] * v[207]) + (v[209] * v[209]);
      v[206] = 1e0 / sqrt(v[381]);
      v[205] = v[204] * v[206];
      v[208] = v[206] * v[207];
      v[210] = v[206] * v[209];
      v[901] = v[108] * v[205];
      v[902] = v[108] * v[208];
      v[903] = v[108] * v[210];
      v[904] = v[109] * v[205];
      v[905] = v[109] * v[208];
      v[906] = v[109] * v[210];
      v[907] = v[110] * v[205];
      v[908] = v[110] * v[208];
      v[909] = v[110] * v[210];
      v[910] = v[111] * v[205];
      v[911] = v[111] * v[208];
      v[912] = v[111] * v[210];
      v[913] = v[112] * v[205];
      v[914] = v[112] * v[208];
      v[915] = v[112] * v[210];
      v[916] = v[113] * v[205];
      v[917] = v[113] * v[208];
      v[918] = v[113] * v[210];
      v[919] = v[114] * v[205];
      v[920] = v[114] * v[208];
      v[921] = v[114] * v[210];
      v[922] = v[115] * v[205];
      v[923] = v[115] * v[208];
      v[924] = v[115] * v[210];
      v[925] = v[116] * v[205];
      v[926] = v[116] * v[208];
      v[927] = v[116] * v[210];
      v[211] = ((v[177] * v[177]) + (v[179] * v[179]) + (v[181] * v[181]) - v[183]) / 2e0;
      v[212] = (v[177] * v[178] + v[179] * v[180] + v[181] * v[182] - v[184]) / 2e0;
      v[247] = -(v[203] * v[214]) - v[202] * v[215] - v[200] * v[216] + v[210] * v[229] +
               v[208] * v[230] + v[205] * v[231];
      v[248] = -(v[203] * v[217]) - v[202] * v[218] - v[200] * v[219] + v[210] * v[235] +
               v[208] * v[236] + v[205] * v[237];
      v[250] = v[165] * v[251];
      v[252] = v[167] * v[251];
      v[259] = -(v[202] * v[250]) + v[200] * v[252];
      v[253] = v[169] * v[251];
      v[257] = v[203] * v[250] - v[200] * v[253];
      v[254] = -(v[203] * v[252]) + v[202] * v[253];
      v[256] = 1e0 / sqrt((v[254] * v[254]) + (v[257] * v[257]) + (v[259] * v[259]));
      v[255] = v[254] * v[256];
      v[258] = v[256] * v[257];
      v[260] = v[256] * v[259];
      v[261] = (v[165] * v[187] + v[166] * v[188]) * v[255] +
               (v[167] * v[187] + v[168] * v[188]) * v[258] +
               (v[169] * v[187] + v[170] * v[188]) * v[260];
      v[269] = (v[261] * v[261]);
      v[262] = (v[165] * v[188] + v[166] * v[189]) * v[255] +
               (v[167] * v[188] + v[168] * v[189]) * v[258] +
               (v[169] * v[188] + v[170] * v[189]) * v[260];
      v[271] = (v[262] * v[262]);
      v[270] = 2e0 * v[261] * v[262];
      v[263] = v[211] * v[266];
      v[265] =
          v[211] * v[269] + v[212] * v[270] +
          (((v[178] * v[178]) + (v[180] * v[180]) + (v[182] * v[182]) - v[185]) * v[271]) / 2e0;
      v[267] = v[247] * v[266];
      v[272] = v[247] * v[269] + v[248] * v[270] +
               (-(v[203] * v[220]) - v[202] * v[221] - v[200] * v[222] + v[210] * v[241] +
                   v[208] * v[242] + v[205] * v[243]) *
                   v[271];
      v[283] = v[263] * v[276] + v[265] * v[277];
      v[284] = v[265] * v[276] + v[263] * v[277];
      v[285] = (v[211] * v[261] + v[212] * v[262]) * v[278] * v[643];
      v[286] = v[267] * v[280] + v[272] * v[281];
      v[287] = v[272] * v[280] + v[267] * v[281];
      v[288] = (v[247] * v[261] + v[248] * v[262]) * v[282] * v[643];
      v[289] = v[251] * v[262];
      v[290] = v[251] * v[261];
      v[291] = v[177] / 2e0;
      v[292] = v[289] * v[291];
      v[293] = v[178] * v[271] + v[270] * v[291];
      v[294] = v[179] / 2e0;
      v[295] = v[289] * v[294];
      v[296] = v[180] * v[271] + v[270] * v[294];
      v[297] = v[181] / 2e0;
      v[298] = v[289] * v[297];
      v[299] = v[182] * v[271] + v[270] * v[297];
      v[300] = v[178] / 2e0;
      v[301] = v[180] / 2e0;
      v[302] = v[182] / 2e0;
      v[303] = v[177] * v[266];
      v[304] = v[177] * v[290] + v[289] * v[300];
      v[305] = v[177] * v[269] + v[270] * v[300];
      v[306] = v[179] * v[266];
      v[307] = v[179] * v[290] + v[289] * v[301];
      v[308] = v[179] * v[269] + v[270] * v[301];
      v[309] = v[181] * v[266];
      v[847] = v[126] * v[303];
      v[848] = v[126] * v[306];
      v[849] = v[126] * v[309];
      v[850] = v[127] * v[303];
      v[851] = v[127] * v[306];
      v[852] = v[127] * v[309];
      v[853] = v[128] * v[303];
      v[854] = v[128] * v[306];
      v[855] = v[128] * v[309];
      v[856] = v[129] * v[303];
      v[857] = v[129] * v[306];
      v[858] = v[129] * v[309];
      v[859] = v[130] * v[303];
      v[860] = v[130] * v[306];
      v[861] = v[130] * v[309];
      v[862] = v[131] * v[303];
      v[863] = v[131] * v[306];
      v[864] = v[131] * v[309];
      v[865] = v[132] * v[303];
      v[866] = v[132] * v[306];
      v[867] = v[132] * v[309];
      v[868] = v[133] * v[303];
      v[869] = v[133] * v[306];
      v[870] = v[133] * v[309];
      v[871] = v[134] * v[303];
      v[872] = v[134] * v[306];
      v[873] = v[134] * v[309];
      v[310] = v[181] * v[290] + v[289] * v[302];
      v[820] = v[135] * v[292] + v[126] * v[304];
      v[821] = v[135] * v[295] + v[126] * v[307];
      v[822] = v[135] * v[298] + v[126] * v[310];
      v[823] = v[136] * v[292] + v[127] * v[304];
      v[824] = v[136] * v[295] + v[127] * v[307];
      v[825] = v[136] * v[298] + v[127] * v[310];
      v[826] = v[137] * v[292] + v[128] * v[304];
      v[827] = v[137] * v[295] + v[128] * v[307];
      v[828] = v[137] * v[298] + v[128] * v[310];
      v[829] = v[138] * v[292] + v[129] * v[304];
      v[830] = v[138] * v[295] + v[129] * v[307];
      v[831] = v[138] * v[298] + v[129] * v[310];
      v[832] = v[139] * v[292] + v[130] * v[304];
      v[833] = v[139] * v[295] + v[130] * v[307];
      v[834] = v[139] * v[298] + v[130] * v[310];
      v[835] = v[140] * v[292] + v[131] * v[304];
      v[836] = v[140] * v[295] + v[131] * v[307];
      v[837] = v[140] * v[298] + v[131] * v[310];
      v[838] = v[141] * v[292] + v[132] * v[304];
      v[839] = v[141] * v[295] + v[132] * v[307];
      v[840] = v[141] * v[298] + v[132] * v[310];
      v[841] = v[142] * v[292] + v[133] * v[304];
      v[842] = v[142] * v[295] + v[133] * v[307];
      v[843] = v[142] * v[298] + v[133] * v[310];
      v[844] = v[143] * v[292] + v[134] * v[304];
      v[845] = v[143] * v[295] + v[134] * v[307];
      v[846] = v[143] * v[298] + v[134] * v[310];
      v[311] = v[181] * v[269] + v[270] * v[302];
      v[955] = v[135] * v[293] + v[126] * v[305];
      v[956] = v[135] * v[296] + v[126] * v[308];
      v[957] = v[135] * v[299] + v[126] * v[311];
      v[958] = v[136] * v[293] + v[127] * v[305];
      v[959] = v[136] * v[296] + v[127] * v[308];
      v[960] = v[136] * v[299] + v[127] * v[311];
      v[961] = v[137] * v[293] + v[128] * v[305];
      v[962] = v[137] * v[296] + v[128] * v[308];
      v[963] = v[137] * v[299] + v[128] * v[311];
      v[964] = v[138] * v[293] + v[129] * v[305];
      v[965] = v[138] * v[296] + v[129] * v[308];
      v[966] = v[138] * v[299] + v[129] * v[311];
      v[967] = v[139] * v[293] + v[130] * v[305];
      v[968] = v[139] * v[296] + v[130] * v[308];
      v[969] = v[139] * v[299] + v[130] * v[311];
      v[970] = v[140] * v[293] + v[131] * v[305];
      v[971] = v[140] * v[296] + v[131] * v[308];
      v[972] = v[140] * v[299] + v[131] * v[311];
      v[973] = v[141] * v[293] + v[132] * v[305];
      v[974] = v[141] * v[296] + v[132] * v[308];
      v[975] = v[141] * v[299] + v[132] * v[311];
      v[976] = v[142] * v[293] + v[133] * v[305];
      v[977] = v[142] * v[296] + v[133] * v[308];
      v[978] = v[142] * v[299] + v[133] * v[311];
      v[979] = v[143] * v[293] + v[134] * v[305];
      v[980] = v[143] * v[296] + v[134] * v[308];
      v[981] = v[143] * v[299] + v[134] * v[311];
      v[793] = v[117] * v[205];
      v[794] = v[117] * v[208];
      v[795] = v[117] * v[210];
      v[796] = v[118] * v[205];
      v[797] = v[118] * v[208];
      v[798] = v[118] * v[210];
      v[799] = v[119] * v[205];
      v[800] = v[119] * v[208];
      v[801] = v[119] * v[210];
      v[802] = v[120] * v[205];
      v[803] = v[120] * v[208];
      v[804] = v[120] * v[210];
      v[805] = v[121] * v[205];
      v[806] = v[121] * v[208];
      v[807] = v[121] * v[210];
      v[808] = v[122] * v[205];
      v[809] = v[122] * v[208];
      v[810] = v[122] * v[210];
      v[811] = v[123] * v[205];
      v[812] = v[123] * v[208];
      v[813] = v[123] * v[210];
      v[814] = v[124] * v[205];
      v[815] = v[124] * v[208];
      v[816] = v[124] * v[210];
      v[817] = v[125] * v[205];
      v[818] = v[125] * v[208];
      v[819] = v[125] * v[210];
      v[342] = v[229] * v[266];
      v[343] = v[235] * v[289] + v[229] * v[290];
      v[344] = v[229] * v[269] + v[235] * v[270] + v[241] * v[271];
      v[345] = v[230] * v[266];
      v[346] = v[236] * v[289] + v[230] * v[290];
      v[347] = v[230] * v[269] + v[236] * v[270] + v[242] * v[271];
      v[348] = v[231] * v[266];
      v[349] = v[237] * v[289] + v[231] * v[290];
      v[350] = v[231] * v[269] + v[237] * v[270] + v[243] * v[271];
      v[766] = v[205] * v[99];
      v[767] = v[208] * v[99];
      v[768] = v[210] * v[99];
      v[769] = v[100] * v[205];
      v[770] = v[100] * v[208];
      v[771] = v[100] * v[210];
      v[772] = v[101] * v[205];
      v[773] = v[101] * v[208];
      v[774] = v[101] * v[210];
      v[775] = v[102] * v[205];
      v[776] = v[102] * v[208];
      v[777] = v[102] * v[210];
      v[778] = v[103] * v[205];
      v[779] = v[103] * v[208];
      v[780] = v[103] * v[210];
      v[781] = v[104] * v[205];
      v[782] = v[104] * v[208];
      v[783] = v[104] * v[210];
      v[784] = v[105] * v[205];
      v[785] = v[105] * v[208];
      v[786] = v[105] * v[210];
      v[787] = v[106] * v[205];
      v[788] = v[106] * v[208];
      v[789] = v[106] * v[210];
      v[790] = v[107] * v[205];
      v[791] = v[107] * v[208];
      v[792] = v[107] * v[210];
      v[378] = v[209] * v[342] + v[207] * v[345] + v[204] * v[348];
      v[379] = v[209] * v[343] + v[207] * v[346] + v[204] * v[349];
      v[380] = v[209] * v[344] + v[207] * v[347] + v[204] * v[350];
      v[382] = -(v[205] / v[381]);
      v[383] = v[206] * v[348] + v[378] * v[382];
      v[384] = v[206] * v[349] + v[379] * v[382];
      v[385] = v[206] * v[350] + v[380] * v[382];
      v[387] = -(v[208] / v[381]);
      v[388] = v[206] * v[345] + v[378] * v[387];
      v[389] = v[206] * v[346] + v[379] * v[387];
      v[390] = v[206] * v[347] + v[380] * v[387];
      v[391] = -(v[210] / v[381]);
      v[392] = v[206] * v[342] + v[378] * v[391];
      v[393] = v[206] * v[343] + v[379] * v[391];
      v[394] = v[206] * v[344] + v[380] * v[391];
      v[395] = v[179] * v[383] - v[177] * v[388];
      v[396] = v[179] * v[384] - v[177] * v[389];
      v[397] = v[179] * v[385] - v[177] * v[390];
      v[398] = -(v[180] * v[383]) + v[178] * v[388];
      v[399] = -(v[180] * v[384]) + v[178] * v[389];
      v[400] = -(v[180] * v[385]) + v[178] * v[390];
      v[401] = v[181] * v[388] - v[179] * v[392];
      v[402] = v[181] * v[389] - v[179] * v[393];
      v[403] = v[181] * v[390] - v[179] * v[394];
      v[404] = -(v[181] * v[383]) + v[177] * v[392];
      v[405] = -(v[181] * v[384]) + v[177] * v[393];
      v[406] = -(v[181] * v[385]) + v[177] * v[394];
      v[407] = -(v[182] * v[388]) + v[180] * v[392];
      v[408] = -(v[182] * v[389]) + v[180] * v[393];
      v[409] = -(v[182] * v[390]) + v[180] * v[394];
      v[410] = v[182] * v[383] - v[178] * v[392];
      v[928] = v[135] * v[401] + v[126] * v[407];
      v[929] = v[135] * v[404] + v[126] * v[410];
      v[930] = v[135] * v[395] + v[126] * v[398];
      v[931] = v[136] * v[401] + v[127] * v[407];
      v[932] = v[136] * v[404] + v[127] * v[410];
      v[933] = v[136] * v[395] + v[127] * v[398];
      v[934] = v[137] * v[401] + v[128] * v[407];
      v[935] = v[137] * v[404] + v[128] * v[410];
      v[936] = v[137] * v[395] + v[128] * v[398];
      v[937] = v[138] * v[401] + v[129] * v[407];
      v[938] = v[138] * v[404] + v[129] * v[410];
      v[939] = v[138] * v[395] + v[129] * v[398];
      v[940] = v[139] * v[401] + v[130] * v[407];
      v[941] = v[139] * v[404] + v[130] * v[410];
      v[942] = v[139] * v[395] + v[130] * v[398];
      v[943] = v[140] * v[401] + v[131] * v[407];
      v[944] = v[140] * v[404] + v[131] * v[410];
      v[945] = v[140] * v[395] + v[131] * v[398];
      v[946] = v[141] * v[401] + v[132] * v[407];
      v[947] = v[141] * v[404] + v[132] * v[410];
      v[948] = v[141] * v[395] + v[132] * v[398];
      v[949] = v[142] * v[401] + v[133] * v[407];
      v[950] = v[142] * v[404] + v[133] * v[410];
      v[951] = v[142] * v[395] + v[133] * v[398];
      v[952] = v[143] * v[401] + v[134] * v[407];
      v[953] = v[143] * v[404] + v[134] * v[410];
      v[954] = v[143] * v[395] + v[134] * v[398];
      v[411] = v[182] * v[384] - v[178] * v[393];
      v[739] = v[135] * v[402] + v[126] * v[408];
      v[740] = v[135] * v[405] + v[126] * v[411];
      v[741] = v[135] * v[396] + v[126] * v[399];
      v[742] = v[136] * v[402] + v[127] * v[408];
      v[743] = v[136] * v[405] + v[127] * v[411];
      v[744] = v[136] * v[396] + v[127] * v[399];
      v[745] = v[137] * v[402] + v[128] * v[408];
      v[746] = v[137] * v[405] + v[128] * v[411];
      v[747] = v[137] * v[396] + v[128] * v[399];
      v[748] = v[138] * v[402] + v[129] * v[408];
      v[749] = v[138] * v[405] + v[129] * v[411];
      v[750] = v[138] * v[396] + v[129] * v[399];
      v[751] = v[139] * v[402] + v[130] * v[408];
      v[752] = v[139] * v[405] + v[130] * v[411];
      v[753] = v[139] * v[396] + v[130] * v[399];
      v[754] = v[140] * v[402] + v[131] * v[408];
      v[755] = v[140] * v[405] + v[131] * v[411];
      v[756] = v[140] * v[396] + v[131] * v[399];
      v[757] = v[141] * v[402] + v[132] * v[408];
      v[758] = v[141] * v[405] + v[132] * v[411];
      v[759] = v[141] * v[396] + v[132] * v[399];
      v[760] = v[142] * v[402] + v[133] * v[408];
      v[761] = v[142] * v[405] + v[133] * v[411];
      v[762] = v[142] * v[396] + v[133] * v[399];
      v[763] = v[143] * v[402] + v[134] * v[408];
      v[764] = v[143] * v[405] + v[134] * v[411];
      v[765] = v[143] * v[396] + v[134] * v[399];
      v[412] = v[182] * v[385] - v[178] * v[394];
      v[874] = v[135] * v[403] + v[126] * v[409];
      v[875] = v[135] * v[406] + v[126] * v[412];
      v[876] = v[135] * v[397] + v[126] * v[400];
      v[877] = v[136] * v[403] + v[127] * v[409];
      v[878] = v[136] * v[406] + v[127] * v[412];
      v[879] = v[136] * v[397] + v[127] * v[400];
      v[880] = v[137] * v[403] + v[128] * v[409];
      v[881] = v[137] * v[406] + v[128] * v[412];
      v[882] = v[137] * v[397] + v[128] * v[400];
      v[883] = v[138] * v[403] + v[129] * v[409];
      v[884] = v[138] * v[406] + v[129] * v[412];
      v[885] = v[138] * v[397] + v[129] * v[400];
      v[886] = v[139] * v[403] + v[130] * v[409];
      v[887] = v[139] * v[406] + v[130] * v[412];
      v[888] = v[139] * v[397] + v[130] * v[400];
      v[889] = v[140] * v[403] + v[131] * v[409];
      v[890] = v[140] * v[406] + v[131] * v[412];
      v[891] = v[140] * v[397] + v[131] * v[400];
      v[892] = v[141] * v[403] + v[132] * v[409];
      v[893] = v[141] * v[406] + v[132] * v[412];
      v[894] = v[141] * v[397] + v[132] * v[400];
      v[895] = v[142] * v[403] + v[133] * v[409];
      v[896] = v[142] * v[406] + v[133] * v[412];
      v[897] = v[142] * v[397] + v[133] * v[400];
      v[898] = v[143] * v[403] + v[134] * v[409];
      v[899] = v[143] * v[406] + v[134] * v[412];
      v[900] = v[143] * v[397] + v[134] * v[400];
      for (i420 = 1; i420 <= 27; i420++)
      {
        v[530] = v[765 + i420];
        v[585] = v[792 + i420];
        res(i420 - 1) += v[418] * v[95] *
                         (2e0 * v[288] * (v[290] * v[530] + v[289] * v[585] + v[738 + i420]) +
                             2e0 * v[285] * v[819 + i420] + v[283] * v[846 + i420] +
                             v[287] * (v[269] * v[530] + v[270] * v[585] + v[873 + i420] +
                                          v[271] * v[900 + i420]) +
                             v[286] * (v[266] * v[530] + v[927 + i420]) + v[284] * v[954 + i420]);
      };
    };
  };
};

FOUR_C_NAMESPACE_CLOSE