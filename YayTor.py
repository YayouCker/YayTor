import math
import scipy.special
import mpmath
import sympy
import matplotlib as plt

class Math:
    # Error function
    def error_function(x):
        return scipy.special.erf(x)

    # Complementary error function
    def complementary_error_function(x):
        return scipy.special.erfc(x)

    # Gamma function
    def gamma_function(x):
        return math.gamma(x)

    # Lower incomplete gamma function
    def lower_incomplete_gamma(a, x):
        return scipy.special.gamma(a, scale=1) * scipy.special.gammainc(a, x)

    # Upper incomplete gamma function
    def upper_incomplete_gamma(a, x):
        return scipy.special.gamma(a, scale=1) * (1 - scipy.special.gammainc(a, x))

    # Digamma function
    def digamma_function(x):
        return scipy.special.psi(x)

    # Trigamma function
    def trigamma_function(x):
        return scipy.special.polygamma(1, x)

    # Beta function
    def beta_function(x, y):
        return math.beta(x, y)

    # Logarithmic integral Li
    def logarithmic_integral_li(x):
        return scipy.special.log1p(x)

    # Exponential integral Ei
    def exponential_integral_ei(x):
        return scipy.special.exp1(x)

    # Sine integral Si
    def sine_integral_si(x):
        return scipy.special.sici(x)[0]

    # Cosine integral Ci
    def cosine_integral_ci(x):
        return scipy.special.sici(x)[1]

    # Airy function Ai
    def airy_function_ai(x):
        return scipy.special.airy(x)[0]

    # Airy function Bi
    def airy_function_bi(x):
        return scipy.special.airy(x)[2]

    # Voigt profile
    def voigt_profile(x, sigma, gamma):
        return scipy.special.voigt_profile(x, sigma, gamma)

    # Riemann zeta function
    def riemann_zeta(x):
        return mpmath.zeta(x)

    # Polygamma function
    def polygamma_function(n, x):
        return mpmath.polygamma(n, x)

    # Regularized incomplete gamma function Q
    def regularized_incomplete_gamma_q(a, x):
        return mpmath.gammainc(a, x)

    # Lambert W function (product logarithm)
    def lambert_w_product_logarithm(x):
        return mpmath.lambertw(x)

    # Lambert W function (K function)
    def lambert_w_k_function(x):
        return mpmath.lambertw(x, k=1)

    # Barnes G-function
    def barnes_g_function(x):
        return mpmath.barnesg(x)

    # Beta function
    def beta_function(x, y):
        return mpmath.beta(x, y)

    # Dirichlet eta function
    def dirichlet_eta_function(x):
        return mpmath.altzeta(x)

    # Polylogarithm function
    def polylogarithm_function(s, z):
        return mpmath.polylog(s, z)

    # Lerch transcendent
    def lerch_transcendent_function(z, s, a):
        return mpmath.lerchphi(z, s, a)

    # Gegenbauer function
    def gegenbauer_function(n, alpha, x):
        return mpmath.gegenbauer(n, alpha, x)

    # Chebyshev function of the first kind
    def chebyshev_function_first_kind(n, x):
        return mpmath.chebyt(n, x)

    # Chebyshev function of the second kind
    def chebyshev_function_second_kind(n, x):
        return mpmath.chebyu(n, x)

    # Legendre function
    def legendre_function(n, x):
        return mpmath.legendre(n, x)

    # Laguerre function
    def laguerre_function(n, alpha, x):
        return mpmath.laguerre(n, alpha, x)

    # Hermite function
    def hermite_function(n, x):
        return mpmath.hermite(n, x)

    # Bessel function J
    def bessel_function_j(n, x):
        return mpmath.besselj(n, x)

    # Bessel function Y
    def bessel_function_y(n, x):
        return mpmath.bessely(n, x)

    # Bessel function I
    def bessel_function_i(n, x):
        return mpmath.besseli(n, x)

    # Bessel function K
    def bessel_function_k(n, x):
        return mpmath.besselk(n, x)

    # Struve function H
    def struve_function_h(v, x):
        return mpmath.struveh(v, x)

    # Weber function E
    def weber_function_e(v, x):
        return mpmath.weber_e(v, x)

    # Parabolic cylinder function D
    def parabolic_cylinder_function_d(v, x):
        return mpmath.pcfd(v, x)

    # Parabolic cylinder function U
    def parabolic_cylinder_function_u(v, x):
        return mpmath.pcfu(v, x)

    # Parabolic cylinder function W
    def parabolic_cylinder_function_w(v, x):
        return mpmath.pcfw(v, x)

    # Tetranacci number
    def tetranacci_number_mpmath(n):
        return mpmath.tetranacci(n)

    # Pentanacci number
    def pentanacci_number_mpmath(n):
        return mpmath.pentanacci(n)

    # Narayana number
    def narayana_number_mpmath(n, k):
        return mpmath.narayana(n, k)

    # Pell number
    def pell_number_mpmath(n):
        return mpmath.pell(n)

    # Companion matrix coefficient
    def companion_matrix_coefficient_mpmath(n):
        return mpmath.companion(n)

    # Bernoulli number
    def bernoulli_number_mpmath(n):
        return mpmath.bernoulli(n)

    # Euler number
    def euler_number_mpmath(n):
        return mpmath.euler(n)

    # Catalan number
    def catalan_number_mpmath(n):
        return mpmath.catalan(n)

    # Gegenbauer polynomial
    def gegenbauer_polynomial_mpmath(n, alpha, x):
        return mpmath.gegenbauer(n, alpha, x)

    # Chebyshev polynomial of the first kind
    def chebyshev_polynomial_first_kind_mpmath(n, x):
        return mpmath.chebyt(n, x)

    # Chebyshev polynomial of the second kind
    def chebyshev_polynomial_second_kind_mpmath(n, x):
        return mpmath.chebyu(n, x)

    # Legendre polynomial
    def legendre_polynomial_mpmath(n, x):
        return mpmath.legendre(n, x)

    # Laguerre polynomial
    def laguerre_polynomial_mpmath(n, alpha, x):
        return mpmath.laguerre(n, alpha, x)

    # Hermite polynomial
    def hermite_polynomial_mpmath(n, x):
        return mpmath.hermite(n, x)

    # Weierstrass elliptic function (℘)
    def weierstrass_p(u, g2, g3):
        return mpmath.ellipj(u, mpmath.mpf(g2), mpmath.mpf(g3))

    # Hyperbolic cotangent function
    def coth(x):
        return 1 / math.tanh(x)

    # Natural logarithm
    def ln(x):
        return math.log(x)

    # Partial derivative
    def partial_derivative(func_form, variable):
        x = sympy.Symbol(variable)
        df = sympy.diff(func_form, x)
        return df

    # Other functions

    # Secant function
    def sec(x):
        return 1 / math.cos(x)

    # Cosecant function
    def csc(x):
        return 1 / math.sin(x)

    # Cotangent function
    def cot(x):
        return 1 / math.tan(x)

    # Hyperbolic secant function
    def sech(x):
        return 1 / math.cosh(x)

    # Hyperbolic cosecant function
    def csch(x):
        return 1 / math.sinh(x)

    # Hyperbolic cotangent function
    def coth(x):
        return 1 / math.tanh(x)

    # Arcsecant function
    def arcsec(x):
        return math.acos(1 / x)

    # Arccosecant function
    def arccsc(x):
        return math.asin(1 / x)

    # Arccotangent function
    def arccot(x):
        return math.atan(1 / x)

    # Hyperbolic arcsecant function
    def arcsech(x):
        return math.acosh(1 / x)

    # Hyperbolic arccosecant function
    def arccsch(x):
        return math.asinh(1 / x)

    # Hyperbolic arccotangent function
    def arccoth(x):
        return math.atanh(1 / x)

    # Step function
    def step(x):
        return 1 if x >= 0 else 0

    # Sign function
    def sign(x):
        if x < 0:
            return -1
        elif x > 0:
            return 1
        else:
            return 0

    # Absolute value function
    def abs_value(x):
        return abs(x)

    # Exponentiation function
    def power(base, exponent):
        return base ** exponent

    # Square root function
    def sqrt(x):
        return math.sqrt(x)

    # Cube root function
    def cbrt(x):
        return x ** (1 / 3)

    # Natural logarithm (base e)
    def ln(x):
        return math.log(x)

    # Logarithm to base 10
    def log10(x):
        return math.log10(x)

    # Logarithm to a specified base
    def log(x, base):
        return math.log(x, base)

    # Floor function (greatest integer less than or equal to x)
    def floor(x):
        return math.floor(x)

    # Ceiling function (least integer greater than or equal to x)
    def ceil(x):
        return math.ceil(x)

    # Truncate function (remove decimal part)
    def truncate(x):
        return int(x)

    # Round to the nearest integer
    def round_to_int(x):
        return round(x)

    # Greatest common divisor (GCD)
    def gcd(a, b):
        return math.gcd(a, b)

    # Least common multiple (LCM)
    def lcm(a, b):
        return a * b // math.gcd(a, b)

    # Factorial
    def factorial(x):
        if x < 0:
            return None
        if x == 0:
            return 1
        result = 1
        for i in range(1, x + 1):
            result *= i
        return result

    # Additional mathematical functions

    # Exponential function
    def exp(x):
        return math.exp(x)

    # Logarithm to a specified base
    def log(x, base):
        return math.log(x, base)

    # Sine integral (Si function)
    def si(x):
        from scipy.special import sici
        si, _ = sici(x)
        return si

    # Cosine integral (Ci function)
    def ci(x):
        from scipy.special import ci
        return ci(x)

    # Error function (Erf function)
    def erf(x):
        from scipy.special import erf
        return erf(x)

    # Complementary error function (Erfc function)
    def erfc(x):
        from scipy.special import erfc
        return erfc(x)

    # Bessel function of the first kind (Jn)
    def bessel_jn(n, x):
        from scipy.special import jn
        return jn(n, x)

    # Bessel function of the second kind (Yn)
    def bessel_yn(n, x):
        from scipy.special import yn
        return yn(n, x)

    # Airy function Ai
    def airy_ai(x):
        from scipy.special import airy
        return airy(x)[0]

    # Airy function Bi
    def airy_bi(x):
        from scipy.special import airy
        return airy(x)[2]

    # Digamma function (ψ)
    def digamma(x):
        from scipy.special import psi
        return psi(x)

    # Zeta function (ζ)
    def zeta(x):
        from scipy.special import zeta
        return zeta(x)

    # Lambert W function (Product Log)
    def lambertw(x):
        from scipy.special import lambertw
        return lambertw(x)

    # Error function for complex numbers
    def complex_error_function(z):
        from scipy.special import wofz
        return wofz(z)

    # Fermi-Dirac distribution
    def fermi_dirac(energy, temperature, is_electron=True):
        k = 8.617333262145 * 10 ** -5  # Boltzmann constant in eV/K
        if is_electron:
            return 1 / (math.exp((energy / (k * temperature)) + 1))
        else:
            return 1 / (math.exp((energy / (k * temperature)) - 1))

    # Fresnel S integral
    def fresnel(x):
        from scipy.special import fresnel
        return fresnel(x)

    # Sine hyperbolic function
    def sinh(x):
        return math.sinh(x)

    # Cosine hyperbolic function
    def cosh(x):
        return math.cosh(x)

    # Tangent hyperbolic function
    def tanh(x):
        return math.tanh(x)

    # Inverse sine hyperbolic function
    def asinh(x):
        return math.asinh(x)

    # Inverse cosine hyperbolic function
    def acosh(x):
        return math.acosh(x)

    # Inverse tangent hyperbolic function
    def atanh(x):
        return math.atanh(x)

    # Lambert W function (real roots)
    def lambertw_real(x):
        from scipy.special import lambertw
        w = lambertw(x)
        if w.real >= 0:
            return w.real
        else:
            return None

    # Exponential integral (Ei function)
    def exponential_integral(x):
        from scipy.special import expi
        return expi(x)

    # Fresnel integral (complex output)
    def fresnel_complex(x):
        from scipy.special import fresnel
        s, c = fresnel(x)
        return s + c * 1j

    # Riemann zeta function (ζ)
    def riemann_zeta(x):
        from scipy.special import zetac
        return zetac(x)

    # Bessel function of the third kind (Hn)
    def bessel_hn(n, x):
        from scipy.special import hankel1, hankel2
        return hankel1(n, x) + 1j * hankel2(n, x)

    # Gudermannian function
    def gudermannian(x):
        return 2 * math.atan(math.tanh(x / 2))

    # Dirac delta function
    def dirac_delta(x):
        if x == 0:
            return float('inf')
        else:
            return 0

    # Kronecker delta function
    def kronecker_delta(i, j):
        return 1 if i == j else 0

    # Polynomial evaluation
    def polynomial(coefficients, x):
        result = 0
        for i, coeff in enumerate(coefficients):
            result += coeff * (x ** i)
        return result

    # Sigmoid function
    def sigmoid(x):
        return 1 / (1 + math.exp(-x))

    # Gompertz function
    def gompertz(x, a, b, c):
        return a * math.exp(-b * math.exp(-c * x))

    # Raised cosine function
    def raised_cosine(x, period=1, duty_cycle=0.5):
        if abs(x) % period < duty_cycle * period:
            return 1
        else:
            return 0

    # Rectangular function
    def rectangular(x, width=1):
        if abs(x) <= width / 2:
            return 1
        else:
            return 0

    # Triangle function
    def triangular(x, width):
        if abs(x) < width / 2:
            return 1 - 2 * abs(x) / width
        else:
            return 0

    # Sinc function
    def sinc(x):
        if x == 0:
            return 1
        else:
            return math.sin(math.pi * x) / (math.pi * x)

    # Hypergeometric function 1F1
    def hypergeometric_1f1(a, b, x):
        from scipy.special import hyp1f1
        return hyp1f1(a, b, x)

    # Incomplete gamma function
    def gamma_incomplete(a, x):
        from scipy.special import gammainc
        return gammainc(a, x)

    # Regularized incomplete beta function
    def beta_incomplete(a, b, x):
        from scipy.special import betainc
        return betainc(a, b, x)

    # Bessel function of the first kind (Jv)
    def bessel_jv(v, x):
        from scipy.special import jv
        return jv(v, x)

    # Bessel function of the second kind (Yv)
    def bessel_yv(v, x):
        from scipy.special import yv
        return yv(v, x)

    # Fractional part function
    def fractional_part(x):
        return x - math.floor(x)

    # Nearest integer function
    def nearest_integer(x):
        return round(x)

    # Sign function
    def sign(x):
        if x > 0:
            return 1
        elif x < 0:
            return -1
        else:
            return 0

    # Softplus function
    def softplus(x):
        return math.log(1 + math.exp(x))

    # Round to the nearest multiple
    def round_to_multiple(x, multiple):
        return round(x / multiple) * multiple

    # Unit step function (Heaviside function)
    def unit_step(x):
        return 1 if x >= 0 else 0

    # Absolute value function
    def absolute_value(x):
        return abs(x)

    # Cube root function
    def cube_root(x):
        return x ** (1 / 3)

    # Floor function
    def floor(x):
        return math.floor(x)

    # Ceiling function
    def ceil(x):
        return math.ceil(x)

    # One's complement (bitwise NOT)
    def ones_complement(x):
        return ~x

    # Square function
    def square(x):
        return x ** 2

    # Cube function
    def cube(x):
        return x ** 3

    # Fourth power function
    def fourth_power(x):
        return x ** 4

    # Square root of the absolute value
    def sqrt_abs(x):
        return math.sqrt(abs(x))

    # Exponential decay function
    def exponential_decay(x, a, b):
        return a * math.exp(-b * x)

    # Ricker wavelet (Mexican hat wavelet)
    def ricker_wavelet(x, sigma):
        return (1 - (x / sigma) ** 2) * math.exp(-x ** 2 / (2 * sigma ** 2))

    # Absolute Gaussian function
    def absolute_gaussian(x, mu, sigma):
        return (1 / (sigma * math.sqrt(2 * math.pi))) * math.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

    # von Mises distribution (circular Gaussian)
    def von_mises(x, mu, kappa):
        from scipy.stats import vonmises
        return vonmises.pdf(x, kappa, loc=mu)

    # Zero-order Bessel function (J0)
    def bessel_j0(x):
        from scipy.special import j0
        return j0(x)

    # Zero-order modified Bessel function of the first kind (I0)
    def bessel_i0(x):
        from scipy.special import i0
        return i0(x)

    # Zero-order Airy function Ai(0)
    def airy_ai_0():
        from scipy.special import airy
        return airy(0)[0]

    # Zero-order Airy function Bi(0)
    def airy_bi_0():
        from scipy.special import airy
        return airy(0)[2]

    # Linear interpolation
    def linear_interpolation(x, x0, x1, y0, y1):
        return y0 + (x - x0) * (y1 - y0) / (x1 - x0)

    def logit(x):
        if x > 0 and x < 1:
            return math.log(x / (1 - x))
        else:
            # Handle out-of-bounds input gracefully (e.g., return None or raise an exception)
            raise ValueError("Input x must be in the open interval (0, 1)")

    def graphplot(list1, list2, xlabel, ylabel, title, plot_type):
        if plot_type == "plot":
            plt.plot(list1, list2)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.title(title)
            plt.show()

        if plot_type == "scatter":
            plt.scatter(list1, list2)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.title(title)
            plt.show()

        if plot_type == "stem":
            plt.stem(list1, list2)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.title(title)
            plt.show()

        if plot_type == "stackplot":
            plt.stackplot(list1, list2)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.title(title)
            plt.show()

class Chemistry:
    def graphplot(list1, list2, xlabel, ylabel, title, plot_type):
        if plot_type == "plot":
            plt.plot(list1, list2)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.title(title)
            plt.show()

        if plot_type == "scatter":
            plt.scatter(list1, list2)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.title(title)
            plt.show()

        if plot_type == "stem":
            plt.stem(list1, list2)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.title(title)
            plt.show()

        if plot_type == "stackplot":
            plt.stackplot(list1, list2)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.title(title)
            plt.show()

    def formula(substance):
        formulas = {
            "acetic acid": "CH3COOH",
            "hydrochloric acid": "HCl",
            "sulfuric acid": "H2SO4",
            "acetate": "CH3COO–",
            "ammonia": "NH3",
            "nitric acid": "HNO3",
            "phosphoric acid": "H3PO4",
            "sodium phosphate": "Na3PO4",
            "calcium carbonate": "CaCO3",
            "ammonium sulfate": "(NH4)2SO4",
            "carbonic acid": "H2CO3",
            "sodium bicarbonate": "NaHCO3",
            "sodium hydroxide": "NaOH",
            "calcium hydroxide": "Ca(OH)2",
            "ethanol": "C2H5OH",
            "hydrobromic acid": "HBr",
            "nitrous acid": "HNO2",
            "potassium hydroxide": "KOH",
            "silver nitrate": "AgNO3",
            "sodium carbonate": "Na2CO3",
            "sodium chloride": "NaCl",
            "cellulose": "(C6H10O5)n",
            "magnesium hydroxide": "Mg(OH)2",
            "methane": "CH4",
            "nitrogen dioxide": "NO2",
            "sodium nitrate": "NaNO3",
            "sulfurous acid": "H2SO3",
            "aluminium sulfate": "Al2(SO4)3",
            "aluminum oxide": "Al2O3",
            "ammonium nitrate": "NH4NO3",
            "ammonium phosphate": "(NH4)3PO4",
            "barium hydroxide": "Ba(OH)2",
            "butane": "C4H10",
            "propane": "C3H8",
            "ethylene": "C2H4",
            "benzene": "C6H6",
            "methanol": "CH3OH",
            "ethylene glycol": "C2H6O2",
            "propylene glycol": "C3H8O2",
            "butanol": "C4H9OH",
            "propionic acid": "C3H6O2",
            "glycerol": "C3H8O3",
            "lactic acid": "C3H6O3",
            "citric acid": "C6H8O7",
            "sodium hypochlorite": "NaClO",
            "sodium benzoate": "C7H5NaO2",
            "potassium sorbate": "C6H7KO2",
            "ascorbic acid": "C6H8O6",
            "sucralose": "C12H19Cl3O8",
            "paracetamol": "C8H9NO2",
            "ibuprofen": "C13H18O2",
            "caffeine": "C8H10N4O2",
            "sodium metabisulfite": "Na2S2O5",
            "sodium nitrite": "NaNO2",
            "sodium thiosulfate": "Na2S2O3",
            "polyethylene": "(C2H4)n",
            "polypropylene": "(C3H6)n",
            "polyvinyl chloride (PVC)": "(C2H3Cl)n",
            "polytetrafluoroethylene (PTFE)": "(C2F4)n",
            "polyethylene terephthalate (PET)": "(C10H8O4)n",
            "polyurethane": "(C16H32N2O2)n",
            "polystyrene": "(C8H8)n",
            "polyvinyl acetate (PVA)": "(C4H6O2)n",
            "polycarbonate": "(C16H14O3)n",
            "polyacrylonitrile (PAN)": "(C3H3N)n",
            "polybutadiene": "(C4H6)n",
            "polyethylene glycol (PEG)": "(C2H4O)n",
            "polymethyl methacrylate (PMMA)": "(C5H8O2)n",
            "polyvinyl alcohol (PVOH)": "(C2H4O)n",
            "polyvinylidene fluoride (PVDF)": "(C2H2F2)n",
            "polypropylene glycol (PPG)": "(C3H6O2)n",
            "polyvinyl chloride (PVC)": "(C2H3Cl)n",
            "polytetrafluoroethylene (PTFE)": "(C2F4)n",
            "polyvinylidene chloride (PVDC)": "(C2H2Cl2)n",
            "polysorbate 80": "C64H124O26",
            "butylated hydroxytoluene (BHT)": "C15H24O",
            "butylated hydroxyanisole (BHA)": "C11H16O2",
            "chlorophyll": "C55H72MgN4O5",
            "chitosan": "(C6H11O4N)n",
            "lignin": "Complex polymer",
            "anthocyanin": "C15H10O6",
            "xanthan gum": "(C35H49O29)n",
            "guanine": "C5H5N5O",
            "adenine": "C5H5N5",
            "cytosine": "C4H5N3O",
            "thymine": "C5H6N2O2",
            "uracil": "C4H4N2O2",
            "nicotine": "C10H14N2",
            "caffeine": "C8H10N4O2",
            "cocaine": "C17H21NO4",
            "tetrahydrocannabinol (THC)": "C21H30O2",
            "morphine": "C17H19NO3",
            "codeine": "C18H21NO3",
            "aspirin": "C9H8O4",
            "vanillin": "C8H8O3",
            "eugenol": "C10H12O2",
            "menthol": "C10H20O",
            "camphor": "C10H16O",
            "salicylic acid": "C7H6O3",
            "acetylsalicylic acid": "C9H8O4",
            "phenolphthalein": "C20H14O4",
            "thymolphthalein": "C28H30O4",
            "methyl orange": "C14H14N3NaO3S",
            "bromothymol blue": "C27H28Br2O5S",
            "phenol red": "C19H14O5S",
            "indigo carmine": "C16H8N2Na2O8S2",
            "malachite green": "C23H25ClN2",
            "crystal violet": "C25H30ClN3",
            "methylene blue": "C16H18ClN3S",
            "iodine": "I2",
            "bromine": "Br2",
            "chlorine": "Cl2",
            "fluorine": "F2",
            "iodide": "I−",
            "bromide": "Br−",
            "chloride": "Cl−",
            "fluoride": "F−",
            "iodine pentafluoride": "IF5",
            "sulfur hexafluoride": "SF6",
            "phosphorus pentachloride": "PCl5",
            "phosphorus trichloride": "PCl3",
            "phosphorus pentabromide": "PBr5",
            "phosphorus tribromide": "PBr3",
            "sulfur dioxide": "SO2",
            "nitrogen monoxide": "NO",
            "nitrogen dioxide": "NO2",
            "nitrous oxide": "N2O",
            "carbon monoxide": "CO",
            "carbon dioxide": "CO2",
            "water": "H2O",
            "hydrogen sulfide": "H2S",
            "ammonia": "NH3",
            "methane": "CH4",
            "ethylene": "C2H4",
            "acetylene": "C2H2",
            "propylene": "C3H6",
            "butene": "C4H8",
            "isobutane": "C4H10",
            "butadiene": "C4H6",
            "pentane": "C5H12",
            "hexane": "C6H14",
            "heptane": "C7H16",
            "octane": "C8H18",
            "nonane": "C9H20",
            "decane": "C10H22",
            "undecane": "C11H24",
            "dodecane": "C12H26",
            "tridecane": "C13H28",
            "tetradecane": "C14H30",
            "pentadecane": "C15H32",
            "hexadecane": "C16H34",
            "heptadecane": "C17H36",
            "octadecane": "C18H38",
            "nonadecane": "C19H40",
            "eicosane": "C20H42",
            "docosane": "C22H46",
            "tricosane": "C23H48",
            "tetracosane": "C24H50",
            "pentacosane": "C25H52",
            "hexacosane": "C26H54",
            "heptacosane": "C27H56",
            "octacosane": "C28H58",
            "nonacosane": "C29H60",
            "triacontane": "C30H62",
            "dotriacontane": "C32H66",
            "tritriacontane": "C33H68",
            "tetratriacontane": "C34H70",
            "pentatriacontane": "C35H72",
            "hexatriacontane": "C36H74",
            "heptatriacontane": "C37H76",
            "octatriacontane": "C38H78",
            "nonatriacontane": "C39H80",
            "tetracontane": "C40H82",
            "dotetracontane": "C42H86",
            "tritetracontane": "C43H88",
            "tetratetracontane": "C44H90",
            "pentatetracontane": "C45H92",
            "hexatetracontane": "C46H94",
            "heptatetracontane": "C47H96",
            "octatetracontane": "C48H98",
            "nonatetracontane": "C49H100",
            "pentacontane": "C50H102",
            "dopentacontane": "C52H106",
            "tripentacontane": "C53H108",
            "tetrapentacontane": "C54H110",
            "pentapentacontane": "C55H112",
            "hexapentacontane": "C56H114",
            "heptapentacontane": "C57H116",
            "octapentacontane": "C58H118",
            "nonapentacontane": "C59H120",
            "hexacontane": "C60H122",
            "dohexacontane": "C62H126",
            "trihexacontane": "C63H128",
            "tetrahexacontane": "C64H130",
            "pentahexacontane": "C65H132",
            "hexahexacontane": "C66H134",
            "heptahexacontane": "C67H136",
            "octahexacontane": "C68H138",
            "nonahexacontane": "C69H140",
            "heptacontane": "C70H142",
            "doheptacontane": "C72H146",
            "triheptacontane": "C73H148",
            "tetraheptacontane": "C74H150",
            "pentaheptacontane": "C75H152",
            "hexaheptacontane": "C76H154",
            "heptaheptacontane": "C77H156",
            "octaheptacontane": "C78H158",
            "nonaheptacontane": "C79H160",
            "octacontane": "C80H162",
            "dooctacontane": "C82H166",
            "trioctacontane": "C83H168",
            "tetraoctacontane": "C84H170",
            "petaoctacontane": "C85H172",
            "hexaoctacontane": "C86H174",
            "heptaoctacontane": "C87H176",
            "octaoctacontane": "C88H178",
            "nonaoctacontane": "C89H180",
            "nonane": "C9H20",
            "decane": "C10H22",
            "undecane": "C11H24",
            "dodecane": "C12H26",
            "tridecane": "C13H28",
            "tetradecane": "C14H30",
            "pentadecane": "C15H32",
            "hexadecane": "C16H34",
            "nickel sulfate": "NiSO4",
            "helium": "He",
            "iodide": "I–",
            "lead ii acetate": "Pb(C2H3O2)2",
            "lithium chloride": "LiCl",
            "phosphate ion": "PO43-",
            "potassium fluoride": "KF",
            "potassium sulfite": "K2SO3",
            "silver carbonate": "Ag2CO3",
            "sodium cyanide": "NaCN",
            "sodium nitride": "Na3N",
            "strontium chloride": "SrCl2",
            "strontium nitrate": "Sr(NO3)2",
            "urea": "CH4N2O",
            "bleach": "NaClO",
            "lithium bromide": "LiBr",
            "aluminum fluoride": "AlF3",
            "barium fluoride": "BaF2",
            "butanoic acid": "C4H8O2",
            "calcium hydride": "CaH2",
            "copper ii carbonate": "CuCO3",
            "fluorine": "F",
            "lithium phosphate": "Li3PO4",
            "glycerol": "C3H8O3",
            "hypobromous acid": "HBrO",
            "hypoiodous acid": "HIO",
            "lead iodide": "PbI2",
            "lithium iodide": "LiI",
            "magnesium oxide": "MgO",
            "urethane": "C3H7NO2",
            "nickel nitrate": "Ni(NO3)2",
            "sodium dichromate": "Na2Cr2O7",
            "tartaric acid": "C4H6O6",
            "zinc iodide": "ZnI2",
            "bromine": "Br",
            "aluminum bromide": "AlBr3",
            "sodium percarbonate": "C2H6Na4O12",
            "nickel acetate": "C4H6O4Ni",
            "sodium thiosulfate": "Na2S2O3",
            "acetaldehyde": "C2H4O",
            "copper sulfate": "CuSO4",
            "mannitol": "C6H14O6",
            "calcium chloride": "CaCl2",
            "monosodium glutamate": "C5H8NO4Na",
            "polystyrene": "(C8H8)n",
            "calcium carbide": "CaC2",
            "tetrachloroethylene": "C2Cl4",
            "sodium chlorate": "NaClO3",
            "potassium iodate": "KIO3",
            "lead acetate": "Pb(C2H3O2)2",
            "potassium thiocyanate": "KSCN",
            # ... continue adding more substances and their formulas ...
        }

        if substance in formulas:
            print(formulas[substance])
        else:
            print("Formula not available for this substance, try changing the capital letters into small letters or check the spelling if needed")

    atomic_masses = {
        'H': 1.008,
        'He': 4.0026,
        'Li': 6.94,
        'Be': 9.0122,
        'B': 10.81,
        'C': 12.011,
        'N': 14.007,
        'O': 16.00,
        'F': 18.998,
        'Ne': 20.180,
        'Na': 22.990,
        'Mg': 24.305,
        'Al': 26.982,
        'Si': 28.085,
        'P': 30.974,
        'S': 32.06,
        'Cl': 35.45,
        'K': 39.098,
        'Ar': 39.948,
        'Ca': 40.078,
        'Sc': 44.956,
        'Ti': 47.867,
        'V': 50.942,
        'Cr': 51.996,
        'Mn': 54.938,
        'Fe': 55.845,
        'Ni': 58.693,
        'Co': 58.933,
        'Cu': 63.546,
        'Zn': 65.38,
        'Ga': 69.723,
        'Ge': 72.630,
        'As': 74.922,
        'Se': 78.971,
        'Br': 79.904,
        'Kr': 83.798,
        'Rb': 85.468,
        'Sr': 87.62,
        'Y': 88.906,
        'Zr': 91.224,
        'Nb': 92.906,
        'Mo': 95.95,
        'Tc': 98.0,
        'Ru': 101.07,
        'Rh': 102.91,
        'Pd': 106.42,
        'Ag': 107.87,
        'Cd': 112.41,
        'In': 114.82,
        'Sn': 118.71,
        'Sb': 121.76,
        'Te': 127.60,
        'I': 126.904,
        'Xe': 131.29,
        'Cs': 132.91,
        'Ba': 137.33,
        'La': 138.91,
        'Ce': 140.12,
        'Pr': 140.91,
        'Nd': 144.24,
        'Pm': 145.0,
        'Sm': 150.36,
        'Eu': 151.96,
        'Gd': 157.25,
        'Tb': 158.93,
        'Dy': 162.50,
        'Ho': 164.93,
        'Er': 167.26,
        'Tm': 168.93,
        'Yb': 173.05,
        'Lu': 174.97,
        'Hf': 178.49,
        'Ta': 180.95,
        'W': 183.84,
        'Re': 186.21,
        'Os': 190.23,
        'Ir': 192.22,
        'Pt': 195.08,
        'Au': 196.97,
        'Hg': 200.59,
        'Tl': 204.38,
        'Pb': 207.2,
        'Bi': 208.98,
        'Th': 232.04,
        'Pa': 231.04,
        'U': 238.03,
        'Np': 237.0,
        'Pu': 244.0,
        'Am': 243.0,
        'Cm': 247.0,
        'Bk': 247.0,
        'Cf': 251.0,
        'Es': 252.0,
        'Fm': 257.0,
        'Md': 258.0,
        'No': 259.0,
        'Lr': 262.0,
    }

    def get_molar_mass(element_symbol):
        return Chemistry.atomic_masses.get(element_symbol, None)

    def get_atomic_mass(substance):
        # Put in the substance like that : [('H', 2), ('O', 1)] (example for H2O)
        mass = 0.0
        for element, count in substance:
            molar_mass = Chemistry.get_molar_mass(element)
            if molar_mass is not None:
                mass += molar_mass * count
            else:
                print(f"Error: Element {element} not found in the database.")
                return None
        return mass
