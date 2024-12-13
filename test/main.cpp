//
// Created by sebas on 17.11.2024.
//
#include <nn/math/matrix.hpp>
#include <nn/math/vector.hpp>
#include <nn/layer/affine.hpp>
#include <nn/layer/relu.hpp>
#include <nn/layer/sigmoid.hpp>
#include <nn/loss/loss.hpp>
#include <numbers>
#include<print>
using namespace math;

Matrix get_input() {
    return {
            {-0.5, -0.40909091, -std::numbers::inv_pi, -0.22727273},
            {-0.13636364, -0.04545455, 0.04545455, 0.13636364},
            {0.22727273, std::numbers::inv_pi, 0.40909091, 0.5}
    };
}

void sigmoid_test() {
    Matrix in = get_input();
    Matrix out_truth = {
        {0.37754067, 0.39913012, 0.42111892, 0.44342513},
        {0.46596182, 0.48863832, 0.51136168, 0.53403818},
        {0.55657487, 0.57888108, 0.60086988, 0.62245933}
    };

    Matrix dx_truth = {
        {-0.11750186, -0.09811034, -0.07756566, -0.05609075},
        {-0.03393292, -0.01135777, 0.01135777, 0.03393292},
        {0.05609075, 0.07756566, 0.09811034, 0.11750186},
    };

    layer::Sigmoid sigmoid = layer::Sigmoid(in.n(), in.m());
    const Matrix &out = sigmoid.forward(in);
    out.print();
    const Matrix &dx = sigmoid.backward(in);
    dx.print();
}

void relu_test() {
    Matrix in = get_input();
    Matrix out_truth = {
        {0.00000000, 0.00000000, 0.00000000, 0.00000000},
        {0.00000000, 0.00000000, 0.04545455, 0.13636364},
        {0.22727273, 0.31818182, 0.40909091, 0.50000000},
    };

    // Matrix dx_truth = out_truth;
    layer::ReLu relu = layer::ReLu(in.n(), in.m());
    const Matrix &out = relu.forward(in);
    out.print();
    const Matrix &dx = relu.backward(in);
    dx.print();
}

void affine_test() {
    Matrix in = {
    {-0.10000000, -0.09748954, -0.09497908, -0.09246862, -0.08995816, -0.08744770, -0.08493724, -0.08242678, -0.07991632, -0.07740586, -0.07489540, -0.07238494, -0.06987448, -0.06736402, -0.06485356, -0.06234310, -0.05983264, -0.05732218, -0.05481172, -0.05230126, -0.04979079, -0.04728033, -0.04476987, -0.04225941, -0.03974895, -0.03723849, -0.03472803, -0.03221757, -0.02970711, -0.02719665, -0.02468619, -0.02217573, -0.01966527, -0.01715481, -0.01464435, -0.01213389, -0.00962343, -0.00711297, -0.00460251, -0.00209205, 0.00041841, 0.00292887, 0.00543933, 0.00794979, 0.01046025, 0.01297071, 0.01548117, 0.01799163, 0.02050209, 0.02301255, 0.02552301, 0.02803347, 0.03054393, 0.03305439, 0.03556485, 0.03807531, 0.04058577, 0.04309623, 0.04560669, 0.04811715, 0.05062762, 0.05313808, 0.05564854, 0.05815900, 0.06066946, 0.06317992, 0.06569038, 0.06820084, 0.07071130, 0.07322176, 0.07573222, 0.07824268, 0.08075314, 0.08326360, 0.08577406, 0.08828452, 0.09079498, 0.09330544, 0.09581590, 0.09832636, 0.10083682, 0.10334728, 0.10585774, 0.10836820, 0.11087866, 0.11338912, 0.11589958, 0.11841004, 0.12092050, 0.12343096, 0.12594142, 0.12845188, 0.13096234, 0.13347280, 0.13598326, 0.13849372, 0.14100418, 0.14351464, 0.14602510, 0.14853556, 0.15104603, 0.15355649, 0.15606695, 0.15857741, 0.16108787, 0.16359833, 0.16610879, 0.16861925, 0.17112971, 0.17364017, 0.17615063, 0.17866109, 0.18117155, 0.18368201, 0.18619247, 0.18870293, 0.19121339, 0.19372385, 0.19623431, 0.19874477},
    {0.20125523, 0.20376569, 0.20627615, 0.20878661, 0.21129707, 0.21380753, 0.21631799, 0.21882845, 0.22133891, 0.22384937, 0.22635983, 0.22887029, 0.23138075, 0.23389121, 0.23640167, 0.23891213, 0.24142259, 0.24393305, 0.24644351, 0.24895397, 0.25146444, 0.25397490, 0.25648536, 0.25899582, 0.26150628, 0.26401674, 0.26652720, 0.26903766, 0.27154812, 0.27405858, 0.27656904, 0.27907950, 0.28158996, 0.28410042, 0.28661088, 0.28912134, 0.29163180, 0.29414226, 0.29665272, 0.29916318, 0.30167364, 0.30418410, 0.30669456, 0.30920502, 0.31171548, 0.31422594, 0.31673640, 0.31924686, 0.32175732, 0.32426778, 0.32677824, 0.32928870, 0.33179916, 0.33430962, 0.33682008, 0.33933054, 0.34184100, 0.34435146, 0.34686192, 0.34937238, 0.35188285, 0.35439331, 0.35690377, 0.35941423, 0.36192469, 0.36443515, 0.36694561, 0.36945607, 0.37196653, 0.37447699, 0.37698745, 0.37949791, 0.38200837, 0.38451883, 0.38702929, 0.38953975, 0.39205021, 0.39456067, 0.39707113, 0.39958159, 0.40209205, 0.40460251, 0.40711297, 0.40962343, 0.41213389, 0.41464435, 0.41715481, 0.41966527, 0.42217573, 0.42468619, 0.42719665, 0.42970711, 0.43221757, 0.43472803, 0.43723849, 0.43974895, 0.44225941, 0.44476987, 0.44728033, 0.44979079, 0.45230126, 0.45481172, 0.45732218, 0.45983264, 0.46234310, 0.46485356, 0.46736402, 0.46987448, 0.47238494, 0.47489540, 0.47740586, 0.47991632, 0.48242678, 0.48493724, 0.48744770, 0.48995816, 0.49246862, 0.49497908, 0.49748954, 0.50000000},
    };
    Matrix weights = {
    {-0.20000000, -0.19860724, -0.19721448},
    {-0.19582173, -0.19442897, -0.19303621},
    {-0.19164345, -0.19025070, -0.18885794},
    {-0.18746518, -0.18607242, -0.18467967},
    {-0.18328691, -0.18189415, -0.18050139},
    {-0.17910864, -0.17771588, -0.17632312},
    {-0.17493036, -0.17353760, -0.17214485},
    {-0.17075209, -0.16935933, -0.16796657},
    {-0.16657382, -0.16518106, -0.16378830},
    {-0.16239554, -0.16100279, -0.15961003},
    {-0.15821727, -0.15682451, -0.15543175},
    {-0.15403900, -0.15264624, -0.15125348},
    {-0.14986072, -0.14846797, -0.14707521},
    {-0.14568245, -0.14428969, -0.14289694},
    {-0.14150418, -0.14011142, -0.13871866},
    {-0.13732591, -0.13593315, -0.13454039},
    {-0.13314763, -0.13175487, -0.13036212},
    {-0.12896936, -0.12757660, -0.12618384},
    {-0.12479109, -0.12339833, -0.12200557},
    {-0.12061281, -0.11922006, -0.11782730},
    {-0.11643454, -0.11504178, -0.11364903},
    {-0.11225627, -0.11086351, -0.10947075},
    {-0.10807799, -0.10668524, -0.10529248},
    {-0.10389972, -0.10250696, -0.10111421},
    {-0.09972145, -0.09832869, -0.09693593},
    {-0.09554318, -0.09415042, -0.09275766},
    {-0.09136490, -0.08997214, -0.08857939},
    {-0.08718663, -0.08579387, -0.08440111},
    {-0.08300836, -0.08161560, -0.08022284},
    {-0.07883008, -0.07743733, -0.07604457},
    {-0.07465181, -0.07325905, -0.07186630},
    {-0.07047354, -0.06908078, -0.06768802},
    {-0.06629526, -0.06490251, -0.06350975},
    {-0.06211699, -0.06072423, -0.05933148},
    {-0.05793872, -0.05654596, -0.05515320},
    {-0.05376045, -0.05236769, -0.05097493},
    {-0.04958217, -0.04818942, -0.04679666},
    {-0.04540390, -0.04401114, -0.04261838},
    {-0.04122563, -0.03983287, -0.03844011},
    {-0.03704735, -0.03565460, -0.03426184},
    {-0.03286908, -0.03147632, -0.03008357},
    {-0.02869081, -0.02729805, -0.02590529},
    {-0.02451253, -0.02311978, -0.02172702},
    {-0.02033426, -0.01894150, -0.01754875},
    {-0.01615599, -0.01476323, -0.01337047},
    {-0.01197772, -0.01058496, -0.00919220},
    {-0.00779944, -0.00640669, -0.00501393},
    {-0.00362117, -0.00222841, -0.00083565},
    {0.00055710, 0.00194986, 0.00334262},
    {0.00473538, 0.00612813, 0.00752089},
    {0.00891365, 0.01030641, 0.01169916},
    {0.01309192, 0.01448468, 0.01587744},
    {0.01727019, 0.01866295, 0.02005571},
    {0.02144847, 0.02284123, 0.02423398},
    {0.02562674, 0.02701950, 0.02841226},
    {0.02980501, 0.03119777, 0.03259053},
    {0.03398329, 0.03537604, 0.03676880},
    {0.03816156, 0.03955432, 0.04094708},
    {0.04233983, 0.04373259, 0.04512535},
    {0.04651811, 0.04791086, 0.04930362},
    {0.05069638, 0.05208914, 0.05348189},
    {0.05487465, 0.05626741, 0.05766017},
    {0.05905292, 0.06044568, 0.06183844},
    {0.06323120, 0.06462396, 0.06601671},
    {0.06740947, 0.06880223, 0.07019499},
    {0.07158774, 0.07298050, 0.07437326},
    {0.07576602, 0.07715877, 0.07855153},
    {0.07994429, 0.08133705, 0.08272981},
    {0.08412256, 0.08551532, 0.08690808},
    {0.08830084, 0.08969359, 0.09108635},
    {0.09247911, 0.09387187, 0.09526462},
    {0.09665738, 0.09805014, 0.09944290},
    {0.10083565, 0.10222841, 0.10362117},
    {0.10501393, 0.10640669, 0.10779944},
    {0.10919220, 0.11058496, 0.11197772},
    {0.11337047, 0.11476323, 0.11615599},
    {0.11754875, 0.11894150, 0.12033426},
    {0.12172702, 0.12311978, 0.12451253},
    {0.12590529, 0.12729805, 0.12869081},
    {0.13008357, 0.13147632, 0.13286908},
    {0.13426184, 0.13565460, 0.13704735},
    {0.13844011, 0.13983287, 0.14122563},
    {0.14261838, 0.14401114, 0.14540390},
    {0.14679666, 0.14818942, 0.14958217},
    {0.15097493, 0.15236769, 0.15376045},
    {0.15515320, 0.15654596, 0.15793872},
    {0.15933148, 0.16072423, 0.16211699},
    {0.16350975, 0.16490251, 0.16629526},
    {0.16768802, 0.16908078, 0.17047354},
    {0.17186630, 0.17325905, 0.17465181},
    {0.17604457, 0.17743733, 0.17883008},
    {0.18022284, 0.18161560, 0.18300836},
    {0.18440111, 0.18579387, 0.18718663},
    {0.18857939, 0.18997214, 0.19136490},
    {0.19275766, 0.19415042, 0.19554318},
    {0.19693593, 0.19832869, 0.19972145},
    {0.20111421, 0.20250696, 0.20389972},
    {0.20529248, 0.20668524, 0.20807799},
    {0.20947075, 0.21086351, 0.21225627},
    {0.21364903, 0.21504178, 0.21643454},
    {0.21782730, 0.21922006, 0.22061281},
    {0.22200557, 0.22339833, 0.22479109},
    {0.22618384, 0.22757660, 0.22896936},
    {0.23036212, 0.23175487, 0.23314763},
    {0.23454039, 0.23593315, 0.23732591},
    {0.23871866, 0.24011142, 0.24150418},
    {0.24289694, 0.24428969, 0.24568245},
    {0.24707521, 0.24846797, 0.24986072},
    {0.25125348, 0.25264624, 0.25403900},
    {0.25543175, 0.25682451, 0.25821727},
    {0.25961003, 0.26100279, 0.26239554},
    {0.26378830, 0.26518106, 0.26657382},
    {0.26796657, 0.26935933, 0.27075209},
    {0.27214485, 0.27353760, 0.27493036},
    {0.27632312, 0.27771588, 0.27910864},
    {0.28050139, 0.28189415, 0.28328691},
    {0.28467967, 0.28607242, 0.28746518},
    {0.28885794, 0.29025070, 0.29164345},
    {0.29303621, 0.29442897, 0.29582173},
    {0.29721448, 0.29860724, 0.30000000},
};
    Vector b = Vector({
        -0.30000000,
        -0.10000000,
        0.10000000,
    });
    Matrix out_truth = {
        {1.49834967, 1.70660132, 1.91485297},
        {3.25553199, 3.51413270, 3.77273342},
    };
    layer::Affine affine = layer::Affine(std::move(weights), std::move(b));
    const Matrix &out = affine.forward(in);
    out.print();
    Matrix dx_truth = {
    {-1.01625006, -0.99485812, -0.97346618, -0.95207424, -0.93068230, -0.90929036, -0.88789843, -0.86650649, -0.84511455, -0.82372261, -0.80233067, -0.78093873, -0.75954680, -0.73815486, -0.71676292, -0.69537098, -0.67397904, -0.65258710, -0.63119516, -0.60980323, -0.58841129, -0.56701935, -0.54562741, -0.52423547, -0.50284353, -0.48145159, -0.46005966, -0.43866772, -0.41727578, -0.39588384, -0.37449190, -0.35309996, -0.33170802, -0.31031609, -0.28892415, -0.26753221, -0.24614027, -0.22474833, -0.20335639, -0.18196445, -0.16057252, -0.13918058, -0.11778864, -0.09639670, -0.07500476, -0.05361282, -0.03222088, -0.01082895, 0.01056299, 0.03195493, 0.05334687, 0.07473881, 0.09613075, 0.11752269, 0.13891462, 0.16030656, 0.18169850, 0.20309044, 0.22448238, 0.24587432, 0.26726626, 0.28865819, 0.31005013, 0.33144207, 0.35283401, 0.37422595, 0.39561789, 0.41700983, 0.43840176, 0.45979370, 0.48118564, 0.50257758, 0.52396952, 0.54536146, 0.56675340, 0.58814533, 0.60953727, 0.63092921, 0.65232115, 0.67371309, 0.69510503, 0.71649697, 0.73788890, 0.75928084, 0.78067278, 0.80206472, 0.82345666, 0.84484860, 0.86624053, 0.88763247, 0.90902441, 0.93041635, 0.95180829, 0.97320023, 0.99459217, 1.01598410, 1.03737604, 1.05876798, 1.08015992, 1.10155186, 1.12294380, 1.14433574, 1.16572767, 1.18711961, 1.20851155, 1.22990349, 1.25129543, 1.27268737, 1.29407931, 1.31547124, 1.33686318, 1.35825512, 1.37964706, 1.40103900, 1.42243094, 1.44382288, 1.46521481, 1.48660675, 1.50799869, 1.52939063},
    {-2.09307628, -2.04902726, -2.00497825, -1.96092923, -1.91688021, -1.87283119, -1.82878218, -1.78473316, -1.74068414, -1.69663513, -1.65258611, -1.60853709, -1.56448807, -1.52043906, -1.47639004, -1.43234102, -1.38829201, -1.34424299, -1.30019397, -1.25614495, -1.21209594, -1.16804692, -1.12399790, -1.07994889, -1.03589987, -0.99185085, -0.94780183, -0.90375282, -0.85970380, -0.81565478, -0.77160576, -0.72755675, -0.68350773, -0.63945871, -0.59540970, -0.55136068, -0.50731166, -0.46326264, -0.41921363, -0.37516461, -0.33111559, -0.28706658, -0.24301756, -0.19896854, -0.15491952, -0.11087051, -0.06682149, -0.02277247, 0.02127654, 0.06532556, 0.10937458, 0.15342360, 0.19747261, 0.24152163, 0.28557065, 0.32961966, 0.37366868, 0.41771770, 0.46176672, 0.50581573, 0.54986475, 0.59391377, 0.63796278, 0.68201180, 0.72606082, 0.77010984, 0.81415885, 0.85820787, 0.90225689, 0.94630590, 0.99035492, 1.03440394, 1.07845296, 1.12250197, 1.16655099, 1.21060001, 1.25464903, 1.29869804, 1.34274706, 1.38679608, 1.43084509, 1.47489411, 1.51894313, 1.56299215, 1.60704116, 1.65109018, 1.69513920, 1.73918821, 1.78323723, 1.82728625, 1.87133527, 1.91538428, 1.95943330, 2.00348232, 2.04753133, 2.09158035, 2.13562937, 2.17967839, 2.22372740, 2.26777642, 2.31182544, 2.35587445, 2.39992347, 2.44397249, 2.48802151, 2.53207052, 2.57611954, 2.62016856, 2.66421757, 2.70826659, 2.75231561, 2.79636463, 2.84041364, 2.88446266, 2.92851168, 2.97256069, 3.01660971, 3.06065873, 3.10470775, 3.14875676},
};
    Matrix dw_truth = {
    {0.50535787, 0.53657745, 0.56779704},
    {0.51729230, 0.54968390, 0.58207550},
    {0.52922673, 0.56279034, 0.59635395},
    {0.54116117, 0.57589679, 0.61063241},
    {0.55309560, 0.58900323, 0.62491087},
    {0.56503003, 0.60210968, 0.63918933},
    {0.57696446, 0.61521612, 0.65346779},
    {0.58889889, 0.62832257, 0.66774625},
    {0.60083332, 0.64142902, 0.68202471},
    {0.61276775, 0.65453546, 0.69630317},
    {0.62470218, 0.66764191, 0.71058163},
    {0.63663661, 0.68074835, 0.72486009},
    {0.64857104, 0.69385480, 0.73913855},
    {0.66050548, 0.70696124, 0.75341701},
    {0.67243991, 0.72006769, 0.76769547},
    {0.68437434, 0.73317413, 0.78197393},
    {0.69630877, 0.74628058, 0.79625239},
    {0.70824320, 0.75938702, 0.81053085},
    {0.72017763, 0.77249347, 0.82480931},
    {0.73211206, 0.78559991, 0.83908777},
    {0.74404649, 0.79870636, 0.85336623},
    {0.75598092, 0.81181280, 0.86764469},
    {0.76791535, 0.82491925, 0.88192315},
    {0.77984978, 0.83802569, 0.89620161},
    {0.79178422, 0.85113214, 0.91048006},
    {0.80371865, 0.86423858, 0.92475852},
    {0.81565308, 0.87734503, 0.93903698},
    {0.82758751, 0.89045147, 0.95331544},
    {0.83952194, 0.90355792, 0.96759390},
    {0.85145637, 0.91666437, 0.98187236},
    {0.86339080, 0.92977081, 0.99615082},
    {0.87532523, 0.94287726, 1.01042928},
    {0.88725966, 0.95598370, 1.02470774},
    {0.89919409, 0.96909015, 1.03898620},
    {0.91112852, 0.98219659, 1.05326466},
    {0.92306296, 0.99530304, 1.06754312},
    {0.93499739, 1.00840948, 1.08182158},
    {0.94693182, 1.02151593, 1.09610004},
    {0.95886625, 1.03462237, 1.11037850},
    {0.97080068, 1.04772882, 1.12465696},
    {0.98273511, 1.06083526, 1.13893542},
    {0.99466954, 1.07394171, 1.15321388},
    {1.00660397, 1.08704815, 1.16749234},
    {1.01853840, 1.10015460, 1.18177080},
    {1.03047283, 1.11326104, 1.19604926},
    {1.04240727, 1.12636749, 1.21032772},
    {1.05434170, 1.13947393, 1.22460618},
    {1.06627613, 1.15258038, 1.23888463},
    {1.07821056, 1.16568682, 1.25316309},
    {1.09014499, 1.17879327, 1.26744155},
    {1.10207942, 1.19189972, 1.28172001},
    {1.11401385, 1.20500616, 1.29599847},
    {1.12594828, 1.21811261, 1.31027693},
    {1.13788271, 1.23121905, 1.32455539},
    {1.14981714, 1.24432550, 1.33883385},
    {1.16175157, 1.25743194, 1.35311231},
    {1.17368601, 1.27053839, 1.36739077},
    {1.18562044, 1.28364483, 1.38166923},
    {1.19755487, 1.29675128, 1.39594769},
    {1.20948930, 1.30985772, 1.41022615},
    {1.22142373, 1.32296417, 1.42450461},
    {1.23335816, 1.33607061, 1.43878307},
    {1.24529259, 1.34917706, 1.45306153},
    {1.25722702, 1.36228350, 1.46733999},
    {1.26916145, 1.37538995, 1.48161845},
    {1.28109588, 1.38849639, 1.49589691},
    {1.29303032, 1.40160284, 1.51017537},
    {1.30496475, 1.41470928, 1.52445383},
    {1.31689918, 1.42781573, 1.53873229},
    {1.32883361, 1.44092217, 1.55301075},
    {1.34076804, 1.45402862, 1.56728920},
    {1.35270247, 1.46713507, 1.58156766},
    {1.36463690, 1.48024151, 1.59584612},
    {1.37657133, 1.49334796, 1.61012458},
    {1.38850576, 1.50645440, 1.62440304},
    {1.40044019, 1.51956085, 1.63868150},
    {1.41237462, 1.53266729, 1.65295996},
    {1.42430906, 1.54577374, 1.66723842},
    {1.43624349, 1.55888018, 1.68151688},
    {1.44817792, 1.57198663, 1.69579534},
    {1.46011235, 1.58509307, 1.71007380},
    {1.47204678, 1.59819952, 1.72435226},
    {1.48398121, 1.61130596, 1.73863072},
    {1.49591564, 1.62441241, 1.75290918},
    {1.50785007, 1.63751885, 1.76718764},
    {1.51978450, 1.65062530, 1.78146610},
    {1.53171893, 1.66373174, 1.79574456},
    {1.54365337, 1.67683819, 1.81002302},
    {1.55558780, 1.68994463, 1.82430148},
    {1.56752223, 1.70305108, 1.83857994},
    {1.57945666, 1.71615752, 1.85285840},
    {1.59139109, 1.72926397, 1.86713686},
    {1.60332552, 1.74237042, 1.88141531},
    {1.61525995, 1.75547686, 1.89569377},
    {1.62719438, 1.76858331, 1.90997223},
    {1.63912881, 1.78168975, 1.92425069},
    {1.65106324, 1.79479620, 1.93852915},
    {1.66299767, 1.80790264, 1.95280761},
    {1.67493211, 1.82100909, 1.96708607},
    {1.68686654, 1.83411553, 1.98136453},
    {1.69880097, 1.84722198, 1.99564299},
    {1.71073540, 1.86032842, 2.00992145},
    {1.72266983, 1.87343487, 2.02419991},
    {1.73460426, 1.88654131, 2.03847837},
    {1.74653869, 1.89964776, 2.05275683},
    {1.75847312, 1.91275420, 2.06703529},
    {1.77040755, 1.92586065, 2.08131375},
    {1.78234198, 1.93896709, 2.09559221},
    {1.79427641, 1.95207354, 2.10987067},
    {1.80621085, 1.96517998, 2.12414913},
    {1.81814528, 1.97828643, 2.13842759},
    {1.83007971, 1.99139287, 2.15270605},
    {1.84201414, 2.00449932, 2.16698451},
    {1.85394857, 2.01760577, 2.18126297},
    {1.86588300, 2.03071221, 2.19554143},
    {1.87781743, 2.04381866, 2.20981988},
    {1.88975186, 2.05692510, 2.22409834},
    {1.90168629, 2.07003155, 2.23837680},
    {1.91362072, 2.08313799, 2.25265526},
    {1.92555516, 2.09624444, 2.26693372},
};
    Vector db_truth = Vector({
        4.75388166,
        5.22073402,
        5.68758639
    });
    Matrix dx = affine.backward(std::move(out_truth));
    dx.print();
    affine.print();
}
void l1_test() {
    Matrix y_out = {
        {0.00000000, 0.25000000, 0.50000000, 0.75000000, 1.00000000},
        {0.00000000, 0.25000000, 0.50000000, 0.75000000, 1.00000000},
        {0.00000000, 0.25000000, 0.50000000, 0.75000000, 1.00000000},
        {0.00000000, 0.25000000, 0.50000000, 0.75000000, 1.00000000},
        {0.00000000, 0.25000000, 0.50000000, 0.75000000, 1.00000000},
    };
    Matrix y_truth = {
        {1.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000},
        {0.00000000, 1.00000000, 0.00000000, 0.00000000, 0.00000000},
        {0.00000000, 0.00000000, 1.00000000, 0.00000000, 0.00000000},
        {0.00000000, 0.00000000, 0.00000000, 1.00000000, 0.00000000},
        {0.00000000, 0.00000000, 0.00000000, 0.00000000, 1.00000000},
    };
    // float loss_truth = 0.5f;
    Matrix dloss_truth = {
        {-1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000},
        {0.00000000, -1.00000000, 1.00000000, 1.00000000, 1.00000000},
        {0.00000000, 1.00000000, -1.00000000, 1.00000000, 1.00000000},
        {0.00000000, 1.00000000, 1.00000000, -1.00000000, 1.00000000},
        {0.00000000, 1.00000000, 1.00000000, 1.00000000, 0.00000000},
    };
    auto results = loss::l1(y_out, y_truth);
    std::println("{}", results.first);
    results.second.print();
}

void l2_test() {
    Matrix y_out = {
        {0.00000000, 0.25000000, 0.50000000, 0.75000000, 1.00000000},
        {0.00000000, 0.25000000, 0.50000000, 0.75000000, 1.00000000},
        {0.00000000, 0.25000000, 0.50000000, 0.75000000, 1.00000000},
        {0.00000000, 0.25000000, 0.50000000, 0.75000000, 1.00000000},
        {0.00000000, 0.25000000, 0.50000000, 0.75000000, 1.00000000},
    };
    Matrix y_truth = {
        {1.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000},
        {0.00000000, 1.00000000, 0.00000000, 0.00000000, 0.00000000},
        {0.00000000, 0.00000000, 1.00000000, 0.00000000, 0.00000000},
        {0.00000000, 0.00000000, 0.00000000, 1.00000000, 0.00000000},
        {0.00000000, 0.00000000, 0.00000000, 0.00000000, 1.00000000},
    };
    // float loss_truth = 0.375f;
    Matrix dloss_truth = {
        {-2.00000000, 0.50000000, 1.00000000, 1.50000000, 2.00000000},
        {0.00000000, -1.50000000, 1.00000000, 1.50000000, 2.00000000},
        {0.00000000, 0.50000000, -1.00000000, 1.50000000, 2.00000000},
        {0.00000000, 0.50000000, 1.00000000, -0.50000000, 2.00000000},
        {0.00000000, 0.50000000, 1.00000000, 1.50000000, 0.00000000},
    };
    loss::LossFunction loss_function = loss::l2;
    auto results = loss_function(y_out, y_truth);
    std::println("{}", results.first);
    results.second.print();

}

void cross_entropy_test() {
    Matrix y_out = {
        {0.00000000, 0.25000000, 0.50000000, 0.75000000, 1.00000000},
        {0.00000000, 0.25000000, 0.50000000, 0.75000000, 1.00000000},
        {0.00000000, 0.25000000, 0.50000000, 0.75000000, 1.00000000},
        {0.00000000, 0.25000000, 0.50000000, 0.75000000, 1.00000000},
        {0.00000000, 0.25000000, 0.50000000, 0.75000000, 1.00000000},
        {0.00000000, 0.25000000, 0.50000000, 0.75000000, 1.00000000},
        {0.00000000, 0.25000000, 0.50000000, 0.75000000, 1.00000000},
        {0.00000000, 0.25000000, 0.50000000, 0.75000000, 1.00000000},
    };
    std::vector<size_t> labels = {0, 1, 2, 3, 4, 3, 2, 1};
    Matrix y_truth = fromLabels(labels, 5);
    // loss = 1.671111984358621
    Matrix dloss = {
        {-0.11074366, 0.01830550, 0.02350473, 0.03018067, 0.03875275},
        {0.01425634, -0.10669450, 0.02350473, 0.03018067, 0.03875275},
        {0.01425634, 0.01830550, -0.10149527, 0.03018067, 0.03875275},
        {0.01425634, 0.01830550, 0.02350473, -0.09481933, 0.03875275},
        {0.01425634, 0.01830550, 0.02350473, 0.03018067, -0.08624725},
        {0.01425634, 0.01830550, 0.02350473, -0.09481933, 0.03875275},
        {0.01425634, 0.01830550, -0.10149527, 0.03018067, 0.03875275},
        {0.01425634, -0.10669450, 0.02350473, 0.03018067, 0.03875275},
    };
    loss::LossFunction loss_function = loss::cross_entropy;
    auto results = loss_function(y_out, y_truth);
    std::println("{}", results.first);
    results.second.print();
}

int main() {
    // sigmoid_test();
    // relu_test();
    // affine_test();
    // l1_test();
    // l2_test();
    cross_entropy_test();
}
