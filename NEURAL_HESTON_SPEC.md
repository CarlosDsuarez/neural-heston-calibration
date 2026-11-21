# Neural Heston Calibration — Especificación de construcción

> **Estado:** borrador v0.1 · 2026-11-21 · Autor: Carlos
> **Stack:** C++20 (núcleo numérico + inferencia) · Python 3.11+ (entrenamiento, datos, análisis)
> **Origen:** reporte de investigación generado con Gemini, **auditado** y corregido con verificación propia (sección 11).
> **Uso previsto:** (1) guía de construcción para ti y para un agente de código (Claude Code / Cursor); (2) base del repositorio público en GitHub.

---

## 0. Cómo usar este documento

1. Las secciones **1–9** son la especificación técnica: léelas en orden y ejecuta las fases de la sección 6 una a una. Cada fase termina en un *gate* medible.
2. La sección **10** cubre la publicación en GitHub. La sección **11** lista qué afirmaciones del reporte original están verificadas, corregidas o sin verificar. **Revísala antes de copiar cualquier cifra al README.**
3. Etiquetas usadas en todo el documento:

| Etiqueta | Significado |
|---|---|
| `[VERIFICADO]` | Revisado en fuente primaria o de catálogo el 2026-11-21 |
| `[OBJETIVO]` | Meta de diseño. **No es un resultado.** Se mide y se reporta, se cumpla o no |
| `[ESTIMADO]` | Cifra del reporte de Gemini, sin medir. Sirve como punto de partida, no como dato |
| `[NO VERIFICADO]` | No pude confirmarla. No citar en público sin comprobarla |
| `[JUICIO]` | Decisión de diseño razonada, discutible |

**Regla de oro:** ninguna cifra de rendimiento entra al README sin salir de un benchmark propio, reproducible y con el hardware declarado. Los resultados negativos también se publican.

---

## 1. Tesis, hipótesis falsables y no-objetivos

### 1.1 Tesis del proyecto

La narrativa popular dice que las redes neuronales aceleran drásticamente la calibración de Heston. Esa afirmación suele medirse contra baselines débiles (optimizadores sin gradiente, código no vectorizado). Cui, del Baño Rollin y Germano (2017) muestran que con función característica estable, gradiente analítico y Levenberg–Marquardt, la calibración clásica ya es muy rápida `[VERIFICADO: abstract del paper; la cifra de speedup ~10× frente a gradiente numérico es del propio paper]`.

Por eso el proyecto **no** se plantea como "la red vence a Heston lento". Se plantea como **benchmark honesto + infraestructura reutilizable**:

1. Construir en C++20 el mejor baseline clásico razonable (COS + Jacobiano analítico + LM).
2. Medir cuánto aportan, *contra ese baseline*, (a) un surrogate de grilla con Sobolev training y (b) un mapeo inverso directo usado como *warm-start*.
3. Dejar el motor (inferencia en Eigen, Jacobiano forward-mode, circuit-breaker) listo para modelos **sin** función característica cerrada, donde el surrogate sí es indispensable (rBergomi, LSV, etc.).

### 1.2 Hipótesis falsables

Cada una se reporta al final como **confirmada / refutada / no concluyente**, con su intervalo de incertidumbre. Los umbrales son `[OBJETIVO]`.

| ID | Hipótesis | Cómo se falsa |
|---|---|---|
| H1 | COS + Jacobiano analítico + LM, 1 núcleo, cadena de ~200 opciones: p99 < 10 ms | Benchmark E0 |
| H2 | Surrogate de grilla + LM (Eigen float32) es ≥ 3× más rápido que E0 en 1 núcleo, con RMSE en IV ≤ 0.05 vol points dentro del dominio | E2 vs E0 |
| H3 | El warm-start inverso reduce las iteraciones LM ≥ 40 % (mediana) frente a inicio frío | E1/E5 vs E0 |
| H4 | En SPX real, el error estructural de Heston domina al error del surrogate por ≥ 10× (RMSE) | Error budget (sección 7.3) |
| H5 | El surrogate *pointwise* (grillas aleatorias) es más lento que COS en Heston estándar | E3 vs E0 |

Nota de coherencia: el reporte original afirma a la vez que la red "no se justifica por latencia" y que el surrogate de grilla es "5–15× más rápido". Esa tensión es justo lo que H1 y H2 resuelven con medición.

### 1.3 No-objetivos

Trading en vivo, opciones americanas, exóticas, tasas estocásticas, GPU, y cualquier afirmación de alpha o rentabilidad. Rough Heston / rBergomi: **solo** una nota de viabilidad (Fase 8), sin implementación.

### 1.4 Recorte si el tiempo se acorta

Se puede recortar la Fase 5 (inverso directo) y la nota rough de la Fase 8. **No** se recorta la Fase 7 (benchmark honesto): es lo que diferencia este proyecto de un tutorial.

---

## 2. Decisiones de arquitectura (ADR)

| # | Decisión | Elección | Descartado | Razón |
|---|---|---|---|---|
| D1 | Pricer de referencia | COS propio en C++20 + CF estable (sección 3) | Carr–Madan FFT; QuantLib como motor | COS evalúa en strikes arbitrarios y reutiliza φ por vencimiento. QuantLib se usa **solo** como oráculo de tests |
| D2 | Calibrador clásico | LM propio 5×5 sobre Eigen, espacio no restringido | SciPy en el camino crítico | Control total de reparametrización y del circuit-breaker |
| D3 | Inferencia neuronal | MLP propio en Eigen, float32, pesos desde `.bin` | LibTorch (huella, ABI); ONNX Runtime como opción | Elimina overhead de runtime y permite fusionar forward + Jacobiano. ONNX Runtime solo para validación cruzada |
| D4 | Entrenamiento | PyTorch en Python | Entrenar en C++ | Ecosistema y rapidez de iteración |
| D5 | Bindings | nanobind | pybind11 | Menor overhead y menor tiempo de compilación `[JUICIO]` |
| D6 | Salida del surrogate | σ_IV sobre **grilla adaptativa por vencimiento** (log-moneyness escalado por √τ) | Grilla rectangular fija; solo pointwise | Evita sobremuestrear strikes ilíquidos en plazos cortos |
| D7 | Comparación con mercado | **Interpolar la salida del surrogate** hacia las cotizaciones (operador lineal fijo, sección 3.9) | Interpolar el mercado hacia la grilla | No altera los datos de mercado; el error de interpolación queda como componente medible del error budget `[JUICIO]` |
| D8 | Datos | Sintético primero; real solo para la Fase 3 (estrato empírico) y la Fase 7 | Depender de datos reales desde el día 1 | Las Fases 1–6 no necesitan datos de pago |
| D9 | Repro | Todo con semilla, hash de git y manifiesto de configuración | Notebooks sueltos | Credibilidad del benchmark |

### Versiones de dependencias

| Dependencia | Versión de referencia | Estado |
|---|---|---|
| Eigen | **5.0.x** (5.0.0 del 30-sep-2025; 5.0.1 vigente en distribuciones; requiere C++14 o superior) | `[VERIFICADO]` — *corrige* el "3.4.1" del reporte original |
| nanobind | 3.1.0 (28-sep-2026); 3.0.0 el 22-ago-2026; requiere Python ≥ 3.10 | `[VERIFICADO]` |
| QuantLib (solo tests) | 1.44 (13-oct-2026) | `[VERIFICADO]` |
| PyTorch, ONNX Runtime, CMake, scikit-build-core | Fijar al iniciar el proyecto (`requirements.lock`) | `[NO VERIFICADO]` — el reporte cita PyTorch 2.9 y ORT 1.24.2 sin fuente |

---

## 3. Matemática de referencia

### 3.1 Modelo (medida neutral al riesgo)

$$dS_t = (r-q)S_t\,dt + \sqrt{v_t}\,S_t\,dW^{(1)}_t,\qquad dv_t = \kappa(\bar v - v_t)\,dt + \sigma\sqrt{v_t}\,dW^{(2)}_t,\qquad d\langle W^{(1)},W^{(2)}\rangle_t=\rho\,dt$$

Parámetros: $\Theta=\{v_0,\kappa,\bar v,\sigma,\rho\}$.

### 3.2 Función característica estable (Little Heston Trap)

Para $X=\ln(S_T/S_0)$ y $\tau=T-t$:

$$\phi_X(u;\tau,\Theta)=\exp\!\Big(iu(r-q)\tau+\tfrac{\kappa\bar v}{\sigma^2}\Big[(\xi-d)\tau-2\ln\tfrac{1-ge^{-d\tau}}{1-g}\Big]+\tfrac{v_0}{\sigma^2}(\xi-d)\tfrac{1-e^{-d\tau}}{1-ge^{-d\tau}}\Big)$$

$$\xi=\kappa-i\rho\sigma u,\qquad d=\sqrt{\xi^2+\sigma^2(u^2+iu)}\ \ (\mathrm{Re}\,d\ge 0),\qquad g=\frac{\xi-d}{\xi+d}$$

Con $\mathrm{Re}\,d\ge0$ se cumple $|g|\le1$, lo que evita los saltos de rama del logaritmo complejo (Albrecher et al., 2007). Cui et al. (2017) usan una representación de este tipo para obtener el gradiente analítico. **Validar siempre contra QuantLib** (test T1.4): si tus fórmulas difieren de las de esta sección en alguna convención, manda el test, no el documento.

### 3.3 Método COS (Fang–Oosterlee)

Con $x=\ln(S_0/K)$ y $\phi(u;x)=e^{iux}\phi_X(u)$:

$$P(K)\approx e^{-r\tau}\,{\sum_{k=0}^{N-1}}'\ \mathrm{Re}\!\Big\{\phi\!\Big(\tfrac{k\pi}{b-a};x\Big)e^{-ik\pi\frac{a}{b-a}}\Big\}\,V_k$$

(la prima indica que el término $k=0$ se pondera por ½). Coeficientes de payoff:

$$V_k^{\text{call}}=\tfrac{2}{b-a}K\big(\chi_k(0,b)-\psi_k(0,b)\big),\qquad V_k^{\text{put}}=\tfrac{2}{b-a}K\big(-\chi_k(a,0)+\psi_k(a,0)\big)$$

$$\chi_k(c,d)=\tfrac{1}{1+\left(\frac{k\pi}{b-a}\right)^2}\Big[\cos\!\big(\tfrac{k\pi(d-a)}{b-a}\big)e^{d}-\cos\!\big(\tfrac{k\pi(c-a)}{b-a}\big)e^{c}+\tfrac{k\pi}{b-a}\big(\sin\!\big(\tfrac{k\pi(d-a)}{b-a}\big)e^{d}-\sin\!\big(\tfrac{k\pi(c-a)}{b-a}\big)e^{c}\big)\Big]$$

$$\psi_k(c,d)=\begin{cases}\frac{b-a}{k\pi}\big[\sin\!\big(\tfrac{k\pi(d-a)}{b-a}\big)-\sin\!\big(\tfrac{k\pi(c-a)}{b-a}\big)\big] & k\neq0\\ d-c & k=0\end{cases}$$

Reglas de implementación:

- $\phi_X(u_k;\tau_j)$ y sus derivadas dependen del **vencimiento**, no del strike: se evalúan una vez por $(k,j)$ y se reutilizan en todos los strikes del vencimiento.
- Usar la **opción OTM** (put si $K<F$, call si $K\ge F$) para evitar cancelación numérica; recuperar la otra por paridad put-call.
- Rango de truncamiento $[a,b]$: empezar con la regla de cumulantes ($L=10$–$12$) y validar convergencia comparando $N$ con $4N$. Para producción, adoptar el criterio de Junike y Pankrashkin (2022). $N=64$–$128$ es el punto de partida habitual; plazos ultracortos pueden necesitar más `[ESTIMADO]`.
- Verificar las fórmulas de $\chi_k,\psi_k$ contra el paper original (Fang & Oosterlee, 2008/09) antes de fijarlas.

### 3.4 Jacobiano analítico respecto a Θ

Por regla de la cadena sobre $\ln\phi$: $\partial\phi/\partial\Theta_j=\phi\cdot\partial\ln\phi/\partial\Theta_j$. Los dos casos más simples sirven de test unitario:

$$\frac{\partial\ln\phi}{\partial v_0}=\frac{\xi-d}{\sigma^2}\cdot\frac{1-e^{-d\tau}}{1-ge^{-d\tau}},\qquad \frac{\partial\ln\phi}{\partial\bar v}=\frac{\kappa}{\sigma^2}\Big[(\xi-d)\tau-2\ln\tfrac{1-ge^{-d\tau}}{1-g}\Big]$$

Las derivadas respecto a $\kappa,\sigma,\rho$ se derivan reutilizando $d$, $g$, $e^{-d\tau}$ y el logaritmo (Cui et al., 2017). **Validación obligatoria:** contra diferencias finitas centradas y contra autodiff (PyTorch con `complex128`) en 1,000 puntos aleatorios.

El paper reporta que el gradiente analítico es ≈10× más rápido que el numérico `[VERIFICADO]`. El reporte de Gemini decía 4–6× `[no coincide con el paper]`. **Mide el tuyo.**

### 3.5 Precio → volatilidad implícita, y residuales

- Inversión a IV con un algoritmo robusto tipo Jäckel ("Let's Be Rational"). Test: ida y vuelta IV→precio→IV con error ≤ 1e-10.
- Sensibilidad implícita sin costo extra: $\partial\sigma_{IV}/\partial\Theta=(\partial P/\partial\Theta)/\mathcal V_{BS}$.
- **Modo recomendado para el bucle LM (sin invertir IV en cada paso):** residual en precio ponderado por la vega de mercado:

$$r_i=w_i\,\frac{P_i^{\text{mod}}(\Theta)-P_i^{\text{mkt}}}{\max(\mathcal V_i^{\text{mkt}},\epsilon)}\ \approx\ w_i\,(\sigma_i^{\text{mod}}-\sigma_i^{\text{mkt}})$$

- Modo IV directo disponible como alternativa para comparar.

### 3.6 Parametrización y Feller

- **No imponer** la condición de Feller $2\kappa\bar v>\sigma^2$ al calibrar SPX: el mercado la viola con frecuencia y la CF estable es válida igualmente `[JUICIO; la violación rutinaria es práctica común, cifras concretas del reporte original NO VERIFICADAS]`.
- Optimizar en espacio no restringido: $z=(\ln v_0,\ln\kappa,\ln\bar v,\ln\sigma,\operatorname{artanh}\rho)$. Jacobiano al espacio $z$ por regla de la cadena: $\partial P/\partial z_j=\partial P/\partial\Theta_j\cdot\partial\Theta_j/\partial z_j$.
- Variante a comparar: $(\sqrt{v_0},\sqrt{\bar v},\kappa\bar v,\sigma,\rho)$, que desacopla parcialmente la deriva de largo plazo de la velocidad de reversión.

### 3.7 Surrogate: arquitectura y pérdida

- MLP $5\to64\times4\to(J\cdot I)$ con activación **suave** (SiLU / Softplus / ELU / tanh). Evitar ReLU: su Jacobiano es discontinuo y empeora la convergencia de LM `[JUICIO]`.
- Entradas: $z$ (sección 3.6) escalado a $[-1,1]$. Salidas: $\sigma_{IV}$ en la grilla (ablación: varianza total $w=\sigma^2\tau$).
- Pérdida de Sobolev (Huge & Savine 2020; Zhang–Amici–Morandotti 2024):

$$\mathcal L=\frac1B\sum_b\Big(\|\hat\sigma(\Theta^{(b)})-\sigma(\Theta^{(b)})\|_2^2+\lambda_{\text{grad}}\|\nabla_\Theta\hat\sigma(\Theta^{(b)})-J_{\text{COS}}(\Theta^{(b)})\|_F^2\Big)+\lambda_{\text{arb}}\mathcal R_{\text{arb}}(\hat\sigma)$$

  Las etiquetas $J_{\text{COS}}$ salen **gratis** del Jacobiano analítico del pricer (sección 3.4) vía la regla de la sección 3.5. Si la red usa $z$ como entrada, transforma las etiquetas de $\partial/\partial\Theta$ a $\partial/\partial z$.
- $\mathcal R_{\text{arb}}$ penaliza violaciones de monotonía calendario y de convexidad (mariposa). **Límite conocido:** penalizar durante el entrenamiento reduce las violaciones, pero no las garantiza en inferencia. La auditoría exacta posterior sigue siendo obligatoria.

### 3.8 Jacobiano forward-mode en el MLP

Con 5 entradas y ~150–200 salidas, el modo **forward** con 5 tangentes es mucho más barato que el reverse (que necesitaría una pasada por salida). Recurrencia:

```
h0 = z;  J0 = I5                       # J_l ∈ R^{d_l × 5}
for l = 1..L:
    a_l = W_l h_{l-1} + b_l
    A_l = W_l J_{l-1}                  # d_l × 5
    h_l = act(a_l);  J_l = diag(act'(a_l)) · A_l
y = W_out h_L + b_out;  J_y = W_out J_L          # (J·I) × 5
dY/dΘ = J_y · (∂z/∂Θ)
```

Coste por capa ≈ $2\,d_l d_{l-1}\cdot 5$ FLOPs adicionales. Test: igualdad con `torch.autograd.functional.jacobian`.

### 3.9 Interpolación de la salida del surrogate hacia las cotizaciones (D7)

La grilla (vencimientos $\tau_j$, log-moneyness $k_i\propto\sqrt{\tau_j}$) **no depende de Θ**. Por tanto, para una cadena de mercado dada, la interpolación (spline en $k$ por vencimiento; lineal en varianza total entre vencimientos adyacentes) es un **operador lineal fijo** $M\in\mathbb R^{n_{\text{quotes}}\times JI}$, que se construye una sola vez por cadena:

$$\hat y_{\text{mkt}}=M\hat y_{\text{grid}},\qquad J_{\text{mkt}}=M\,J_{\text{grid}}$$

El Jacobiano pasa por la interpolación sin costo extra. Cotizaciones fuera del rango de la grilla se **excluyen y se cuentan** (métrica de cobertura). El error de interpolación se mide como componente propio del error budget.

### 3.10 No-arbitraje estático (auditoría)

Con $w(k,\tau)=\sigma_{IV}^2\tau$ y $k=\ln(K/F)$:

- **Calendario:** $\partial_\tau w(k,\tau)\ge0$ a $k$ fijo.
- **Mariposa (Durrleman / Gatheral–Jacquier):** $g(k)=\big(1-\tfrac{k\,w'}{2w}\big)^2-\tfrac{w'^2}{4}\big(\tfrac1w+\tfrac14\big)+\tfrac{w''}{2}\ge0$.

Se evalúan sobre una malla densa (p. ej. 40 puntos de $k$ por vencimiento) tanto en la salida cruda del surrogate como en la superficie re-valuada con COS.

### 3.11 Inverso directo y warm-start

El mapeo superficie→Θ es **mal condicionado**: κ, v̄ y σ son débilmente identificables, sobre todo con pocos puntos o plazos cortos. Una red determinista entrenada con MSE sobre Θ predice la media condicional, que no reproduce la superficie por la no linealidad del precio. Consecuencias de diseño:

1. El inverso **no** es el calibrador final: es un generador de semilla $\Theta_0$.
2. Siempre se refina con LM y se audita con COS.
3. Comparar dos pérdidas: (a) MSE en Θ; (b) pérdida de **reconstrucción** en espacio de superficie a través del surrogate congelado.
4. Ensemble de 5 semillas: su varianza sirve como señal de incertidumbre/OOD.

---

## 4. Arquitectura del repositorio

```
neural-heston-calibration/
├── CMakeLists.txt
├── pyproject.toml                    # scikit-build-core + nanobind
├── LICENSE  · README.md  · CITATION.cff  · NEURAL_HESTON_SPEC.md
├── cpp/
│   ├── include/nhc/
│   │   ├── types.hpp                 # HestonParams, Market, Quote, Chain
│   │   ├── heston_cf.hpp             # φ estable + ∂lnφ/∂Θ
│   │   ├── cos_pricer.hpp            # COS vectorizado (SoA, alignas(64))
│   │   ├── iv_solver.hpp             # Black IV (Jäckel) + vega
│   │   ├── lm_solver.hpp             # LM 5×5 (amortiguamiento de Nielsen)
│   │   ├── grid.hpp                  # grilla adaptativa + operador M (3.9)
│   │   ├── mlp_eigen.hpp             # inferencia + Jacobiano forward-mode
│   │   ├── ood.hpp                   # circuit-breaker
│   │   └── calibrate.hpp             # cos_lm / surrogate_lm / hybrid
│   ├── src/bindings.cpp              # módulo nanobind `nhc_core`
│   └── tests/                        # GoogleTest/Catch2
├── python/nhc/
│   ├── data/synthetic.py             # muestreo Sobol estratificado + dataset
│   ├── data/spx_ingest.py            # limpieza, forward implícito, filtros
│   ├── train/surrogate.py            # baseline vs Sobolev
│   ├── train/inverse.py              # inverso + ensemble
│   ├── export/to_bin.py              # pesos .bin + metadatos JSON
│   ├── eval/noarb.py                 # auditoría calendario/mariposa
│   ├── bench/run.py                  # E0–E5 + baseline ingenuo
│   └── bench/report.py               # tablas y figuras → results/
├── tests/                            # pytest: paridad, QuantLib, regresión
├── docs/                             # notas, rough_viability.md
├── data/                             # **en .gitignore**
└── results/                          # solo artefactos pequeños y con hardware stamp
```

### 4.1 Interfaces C++ (esqueleto)

```cpp
namespace nhc {
struct HestonParams { double v0, kappa, vbar, sigma, rho; };
struct Market { double S0, r, q; };                              // q: dividendo continuo
struct Quote  { double tau, strike, mid, iv, vega, weight; bool is_call; };
using Chain = std::vector<Quote>;                                 // agrupado por tau ascendente

struct CosConfig { int N = 128; double L = 12.0; };
void price_chain(const HestonParams&, const Market&, const Chain&,
                 const CosConfig&, double* prices);
void price_chain_with_jacobian(const HestonParams&, const Market&, const Chain&,
                 const CosConfig&, double* prices, double* J /* n×5, row-major */);

struct LmConfig { int max_iter = 50; double tol_grad = 1e-8, tol_step = 1e-10, lambda0 = 1e-3; };
struct LmResult { HestonParams theta; double rmse_iv; int iters; bool converged; };
LmResult calibrate_cos_lm(const Market&, const Chain&, HestonParams init,
                          const LmConfig&, const CosConfig&);
LmResult calibrate_surrogate_lm(const Market&, const Chain&, const MlpSurrogate&,
                                HestonParams init, const LmConfig&);
LmResult calibrate_hybrid(const Market&, const Chain&, const MlpInverse&,
                          const LmConfig&, const CosConfig&);  // warm-start + 3–5 its COS
}
```

### 4.2 Formato de pesos `.bin` (versionado)

Cabecera: magic `NHC1` · versión · nº de capas · dimensiones · id de activación · transformaciones de entrada/salida · **metadatos de grilla** (lista de $\tau_j$, regla de $k$) · hash de configuración de entrenamiento y commit de git. Sidecar `.json` con lo mismo en texto. Un cambio de grilla sin cambiar la versión debe **fallar al cargar**.

---

## 5. Datos

### 5.1 Sintéticos (Fases 3–6)

- Muestreo de Θ con secuencias de Sobol, en tres estratos. **Rangos iniciales (`[ESTIMADO]`, ajustar con evidencia propia):**

| Estrato | Peso | Rangos de partida |
|---|---|---|
| Normal / moderadamente elevado | 65 % | $v_0\in[0.005,0.25]$, $\bar v\in[0.01,0.25]$, $\kappa\in[0.2,8]$, $\sigma\in[0.1,2.5]$, $\rho\in[-0.98,-0.15]$ |
| Estrés y colas | 25 % | $v_0\in[0.25,1.0]$, $\bar v\in[0.10,0.50]$, $\sigma\in[1.5,4.5]$, $\kappa\in[0.05,12]$ |
| Empírico perturbado | 10 % | Θ calibrados con COS+LM sobre días reales **de entrenamiento** + perturbación log-normal |

- **Dependencia a respetar:** el estrato empírico necesita datos reales y el calibrador clásico (Fase 2). Por eso la ingesta de SPX va en la Fase 3.
- **Sin fuga temporal:** partición por **fecha** (entrenamiento ≤ T0 < validación ≤ T1 < prueba). El estrato empírico se construye solo con fechas de entrenamiento.
- Entradas: log de los cuatro parámetros positivos. Tamaños para la curva de aprendizaje: 10k, 30k, 60k, 120k, 240k superficies.
- Grilla de partida (`[ESTIMADO]`): ~10 vencimientos entre 0.02 y 2.0 años; 15–20 puntos de $k\in[-0.55\sqrt\tau,+0.35\sqrt\tau]$. Ajustar al soporte real de cotizaciones SPX y a los plazos largos.

### 5.2 Mercado real: SPX europeo

Usar **SPX** (europeo, liquidación en efectivo) como subyacente principal.

**Limpieza por fecha y vencimiento:**

1. Forward y descuento implícitos por regresión de paridad put-call cerca del ATM: $C-P=a-D\,K$, por lo que $D=-\text{pendiente}$ y $F=a/D$.
2. Filtros: bid > 0, ask > bid, spread relativo ≤ 20 % (parametrizable), liquidez mínima (volumen u open interest), $\tau\ge7/365$.
3. Solo OTM (put si $K<F$, call si $K\ge F$).
4. Pesos: inverso del spread en IV, $w_i=1/\max(\text{spread}_{IV,i},0.001)$, o ponderación por vega.
5. Registrar cuántas cotizaciones se descartan por regla (transparencia).

**Fuentes (estado verificado el 2026-11-21):**

| Fuente | Qué ofrece | Estado |
|---|---|---|
| Cboe delayed quotes (endpoints JSON públicos en `cdn.cboe.com`, sin autenticación) | Cadena SPX con retraso, gratis | `[VERIFICADO]` que existen como endpoints públicos según un catálogo de APIs. **URL exacta y términos de uso: verificar** antes de automatizar |
| Alpaca | Opciones sobre índices (SPX, SPXW, VIX, XSP, DJX) disponibles en la Trading API, con paper trading y liquidación en efectivo. **Pero** Alpaca declara que no ofrece datos del índice en su Market Data. Datos históricos de opciones solo desde feb-2024 | `[VERIFICADO]` para trading e histórico. **Si expone cotizaciones de opciones SPX utilizables para calibrar: no confirmado** |
| HistoricalOptionData.com (ALLSPX, EOD desde 1990) | Histórico EOD SPX en CSV | `[VERIFICADO]` precio listado: USD 805. Licencia comercial: leer términos |
| Cboe DataShop | EOD con IV y Greeks a las 15:45 ET | `[VERIFICADO]` que existe. Precio: no confirmado |
| OptionMetrics (WRDS) | Estándar académico | `[VERIFICADO]` que Cboe EOD figura en WRDS. **Depende de que Icesi tenga acceso** |
| OptionsDX, datasets de Kaggle | Históricos SPX gratuitos o de bajo costo | OptionsDX existe como referencia comunitaria; **el dataset de Kaggle citado por Gemini: `[NO VERIFICADO]`** |
| Precios de IBKR, Databento y DataShop que cita el reporte | USD 1.50/mes, 199/mes, 86/mes | `[NO VERIFICADO]`. No usarlos para decidir sin comprobar |

**Plan realista sin presupuesto:** capturar snapshots diarios propios desde los endpoints delayed de Cboe (tras comprobar términos) para el periodo reciente, y complementar con un histórico gratuito si logras uno confiable. Para episodios de estrés pasados, comprar 15–30 días puntuales es más sensato que una suscripción completa.

**Advertencia SPY / ETFs:** sus opciones son **americanas** con dividendos discretos. Si las usas como respaldo, aplica de-americanización (restar la prima de ejercicio anticipado con árbol binomial o PDE con calendario de dividendos), usa solo OTM líquidas y documenta el error residual. Las cifras concretas del reporte sobre SPY vs SPX (dividendo implícito mediano, % de días negativos) son `[NO VERIFICADO]`.

### 5.3 Política de datos para un repo público

- **No subir datos de proveedores.** Casi todas las licencias prohíben redistribuir. `data/` va en `.gitignore`.
- Incluir scripts de descarga/ingesta + un **mini dataset sintético** (≤ 1 MB) para que los tests y `make repro-small` funcionen sin cuenta de nadie.
- Los resultados publicados deben ser agregados (RMSE, latencias, tasas), no cotizaciones crudas.

---

## 6. Plan por fases (8 semanas indicativas)

> Las semanas son orientativas (el reporte original asumía ~15–20 h/semana, `[ESTIMADO]`). Lo que manda son los **gates**. Todos los umbrales son `[OBJETIVO]`: si un gate no se cumple, se reporta el valor real y se decide si iterar o ajustar la meta, sin esconderlo.

### Fase 0 — Fundaciones (días 1–3)
- [ ] T0.1 Repo privado `neural-heston-calibration`; licencia MIT (ver sección 10); `.gitignore` (data/, build/, `*.bin` grandes).
- [ ] T0.2 CMake ≥ 3.26, C++20. Flags: `-O3` y `-fno-math-errno`; **sin** `-ffast-math` en `cos_pricer`/`heston_cf` (el manejo de ramas complejas es delicado); `-ffast-math` solo en `mlp_eigen` si un test de paridad lo permite. Eigen 5.0.x y nanobind vía FetchContent/gestor con versión fijada.
- [ ] T0.3 Entorno Python con lockfile (PyTorch, NumPy, pandas, SciPy, pytest, QuantLib de PyPI solo para tests).
- [ ] T0.4 CI en GitHub Actions: build + `ctest` + `pytest` (gcc y clang en Linux), `clang-format` y `ruff`.
- **Gate G0:** el pipeline completo corre verde en CI con un test trivial.

### Fase 1 — Pricer clásico (semana 1)
- [ ] T1.1 CF estable (3.2) en `std::complex<double>`; nodos $u_k$ en arreglos SoA alineados a 64 bytes.
- [ ] T1.2 COS call/put con OTM + paridad; $N$ y $L$ configurables.
- [ ] T1.3 Derivadas $\partial\ln\phi/\partial\Theta$ y Jacobiano de precios.
- [ ] T1.4 Tests: (a) precios vs `AnalyticHestonEngine` de QuantLib en 1,000 casos aleatorios, **incluyendo Feller violado** y $\tau\in[0.02,3]$; (b) convergencia $N=64\to128\to256\to512$; (c) Jacobiano vs diferencias finitas centradas.
- **Gate G1:** error de precio máximo ≤ 1e-8 (normalizado por $S_0$) en la región normal; error relativo del Jacobiano ≤ 1e-6. Documentar las regiones donde no se cumple (plazos ultracortos, colas).

### Fase 2 — Calibración clásica de punta a punta (semana 2)
- [ ] T2.1 Inversor de IV con test de ida y vuelta.
- [ ] T2.2 LM 5×5 en espacio $z$, Jacobiano por regla de la cadena; modos precio-ponderado-por-vega e IV.
- [ ] T2.3 Bindings nanobind (`price_chain`, `calibrate_cos_lm`).
- [ ] T2.4 **Baseline ingenuo** (reproduce "lo que dice la literatura"): QuantLib en Python + SciPy `least_squares`/Differential Evolution.
- [ ] T2.5 Arnés de benchmark v0 (CPU fija, calentamiento, p50/p95/p99).
- **Gate G2:** sobre 100 superficies sintéticas sin ruido, ≥ 99 % de recuperación de superficie con RMSE en IV ≤ 0.005 vol points desde un inicio frío fijo. Reportar la latencia medida de H1 **sea cual sea**.
- **Hito público:** tras G2 se puede publicar el repo con el baseline clásico, sin ninguna afirmación neuronal.

### Fase 3 — Datos (semana 3)
- [ ] T3.1 `grid.hpp`: grilla adaptativa + construcción del operador $M$ (3.9).
- [ ] T3.2 Ingesta SPX (5.2) y calibración COS+LM de un conjunto de días de **entrenamiento** para el estrato empírico.
- [ ] T3.3 Generador paralelo (OpenMP, `schedule(static)`) de superficies + etiquetas de Jacobiano en σ_IV.
- [ ] T3.4 Formato de dataset (Parquet/NPZ) + manifiesto (semilla, hash de git, configuración).
- **Gate G3:** dataset de 120k superficies con Jacobianos generado; 0 NaN/Inf; error de ida y vuelta de IV ≤ 1e-8; tiempo de generación medido y reportado.

### Fase 4 — Surrogate (semana 4)
- [ ] T4.1 Baseline MLP (5 → 64×4 → grilla) con pérdida en vol points.
- [ ] T4.2 Variante Sobolev (barrido de $\lambda_{\text{grad}}$) y penalización de no-arbitraje opcional.
- [ ] T4.3 **Curva de aprendizaje** (10k…240k) con ≥ 5 semillas por punto; reportar media ± desviación.
- [ ] T4.4 Barrido corto de ancho {32, 64, 128}, profundidad {3, 4, 5}, activación {SiLU, Softplus, ELU, tanh}.
- **Gate G4:** RMSE en IV ≤ 0.05 vol points y p99 ≤ 0.20 en dominio; Sobolev reduce el error relativo del Jacobiano ≥ 3× frente al baseline. Si no se cumple: iterar como máximo una semana y publicar el valor real.

### Fase 5 — Inverso y ablación pointwise (semana 5) *(recortable)*
- [ ] T5.1 Red inversa con las dos pérdidas de 3.11; ensemble de 5 semillas.
- [ ] T5.2 Ablación pointwise con grillas aleatorias (entrada de 7 dimensiones: Θ, τ, k → 1 salida; ver Baschetti et al. 2024).
- [ ] T5.3 Exportador `.bin` + JSON (4.2).
- **Gate G5:** error de reconstrucción **medido con COS exacto** sobre Θ̂ del inverso; el warm-start reduce las iteraciones LM ≥ 40 % (mediana) o se reporta que no (H3).

### Fase 6 — Inferencia en C++ (semana 6)
- [ ] T6.1 `MlpSurrogate` en Eigen (float32) con forward + Jacobiano forward-mode (3.8). Paridad con PyTorch: ≤ 1e-9 en ruta float64 y ≤ 1e-5 relativo en float32.
- [ ] T6.2 `calibrate_surrogate_lm` con el operador $M$ (3.9).
- [ ] T6.3 Circuit-breaker (sección 8) y `calibrate_hybrid`.
- **Gate G6:** latencias medidas p50/p99; el 100 % de los casos OOD sintéticos dispara el fallback; el re-pricing exacto con COS del resultado final está siempre disponible.

### Fase 7 — Benchmark sobre SPX real (semana 7) — *no recortable*
- [ ] T7.1 Partición temporal (≥ 1 episodio de estrés en el conjunto de prueba, si hay datos).
- [ ] T7.2 Ejecutar E0–E5 y el baseline ingenuo (sección 7).
- [ ] T7.3 Tablas y figuras desde un script reproducible (`bench/report.py`).
- [ ] T7.4 Veredicto H1–H5.
- **Gate G7:** `results/` generado por un solo comando; todas las hipótesis con veredicto; hardware y flags de compilación declarados.

### Fase 8 — Cierre y publicación (semana 8)
- [ ] T8.1 `docs/rough_viability.md` (≤ 2 páginas): qué cambia en rBergomi / Rough Heston (sin CF cerrada, Monte Carlo o Riccati fraccionaria), por qué ahí el surrogate sí es necesario, y la evidencia empírica en contra de las ventajas de rough sobre SPX (Abi Jaber & Li).
- [ ] T8.2 README en inglés, `CITATION.cff`, `make repro-small`, release `v0.1.0`.
- **Gate G8:** desde un clon limpio, `make repro-small` reproduce la tabla principal en < 30 min.

---

## 7. Protocolo de benchmark

### 7.1 Motores

| ID | Motor |
|---|---|
| E0 | COS + Jacobiano analítico + LM, **inicio frío** |
| E1 | COS + LM con **warm-start** del inverso |
| E2 | Surrogate de grilla (baseline y Sobolev) + LM |
| E3 | Surrogate pointwise de grillas aleatorias + LM (ablación) |
| E4 | Inverso directo solo |
| E5 | **Híbrido:** inverso → 3–5 iteraciones COS+LM |
| Ref | Baseline ingenuo: QuantLib (Python) + SciPy |

### 7.2 Métricas

1. **Precisión:** RMSE, MAE y p99 del error absoluto en vol points, ponderado y no ponderado, por plazo (< 1 mes, 1–6 meses, > 6 meses) y por moneyness (alas put, ATM, alas call). **Siempre re-valuar el Θ̂ final con COS exacto** y medir el ajuste real. El RMSE del propio surrogate engaña.
2. **Latencia p50/p95/p99** en C++ con `std::chrono::steady_clock`, ≥ 1,000 corridas tras calentamiento, desglosada (interpolación, inferencia, pasos LM, verificación COS). Declarar CPU, frecuencia, compilador y flags. Fijar afinidad de hilo.
3. **Convergencia y sensibilidad al inicio:** fracción que converge sin tocar cotas; 10 inicializaciones dispersas por superficie para exponer mínimos locales.
4. **Estabilidad intertemporal:** $\|\Theta_t-\Theta_{t-1}\|_1$ y, como extensión, rotación de delta y vega entre fechas consecutivas. Dos calibraciones con igual RMSE pueden saltar entre valles planos distintos y generar rotación artificial de coberturas. Sensibilidad opcional con regularización $\lambda_{\text{time}}\|\Theta_t-\Theta_{t-1}\|^2$.
5. **No-arbitraje (3.10):** % de nodos que violan calendario o mariposa por más de 1e-6, en la salida cruda del surrogate y tras COS.

### 7.3 Error budget (para H4)

Descomponer el error final contra el mercado en: (a) error del surrogate, (b) error de interpolación $M$, (c) error del optimizador/mínimos locales, (d) error estructural de Heston vs SPX, (e) ruido bid-ask/desfase de snapshot. Reportar cuál domina. **No hay cifras de partida confiables:** las del reporte original (p. ej. 1.99–2.87 vol points de RMSE estructural, 0.46 en un día suave) son `[NO VERIFICADO]`. Se miden.

### 7.4 Rigor estadístico

- ≥ 5 semillas de entrenamiento; reportar media ± desviación.
- Comparaciones por fecha (pareadas): test de Wilcoxon sobre diferencias de RMSE diario.
- Un solo hardware de referencia, declarado; si hay segundo equipo, tabla aparte.

### 7.5 Plantilla de resultados (se rellena con datos propios)

| Motor | RMSE IV (vol pts) | p50 (ms) | p99 (ms) | Convergencia | Iter. LM (mediana) | Violaciones no-arb |
|---|---|---|---|---|---|---|
| Ref (ingenuo) | — | — | — | — | — | — |
| E0 COS+LM frío | — | — | — | — | — | — |
| E1 COS+LM warm | — | — | — | — | — | — |
| E2 Surrogate grilla | — | — | — | — | — | — |
| E3 Pointwise | — | — | — | — | — | — |
| E4 Inverso solo | — | — | — | — | — | — |
| E5 Híbrido | — | — | — | — | — | — |

### 7.6 Objetivos de latencia de diseño (todos `[ESTIMADO]` por Gemini; la sección 7.5 los reemplaza por valores medidos)

Cadena de 200 opciones, $N_{\text{cos}}=128$, 1 núcleo: COS+LM p50 ≈ 3.5 ms (E0, 1 núcleo); surrogate de grilla + LM p50 ≈ 0.4 ms; inverso solo p50 ≈ 5 µs; híbrido p50 ≈ 1.5 ms. Estos números salen de un conteo de FLOPs con supuestos de GFLOP/s sin medir. Úsalos solo para saber si algo está *órdenes de magnitud* fuera de lugar.

---

## 8. Riesgos y circuit-breaker

### 8.1 Especificación del circuit-breaker

Se evalúa antes y después de cada calibración neuronal. Si **cualquier** condición se cumple, se cambia (en silencio y con log) a COS+LM:

1. Algún componente de $z$ entra en el margen exterior (p. ej. 5 %) de la caja de entrenamiento.
2. Distancia de Mahalanobis de la superficie de entrada (o varianza del ensemble inverso) sobre un umbral calibrado en validación.
3. El residuo reconstruido con COS exacto difiere del residuo del surrogate en más de 0.15 vol points `[OBJETIVO]`.
4. LM no converge o toca cotas.
5. La superficie de mercado queda fuera del soporte entrenado (p. ej. IV ATM por encima del rango del dataset).

Todos los umbrales se calibran sobre datos de validación y se reportan.

### 8.2 Matriz de riesgos

| Riesgo | Mecanismo | Efecto de segundo orden | Mitigación |
|---|---|---|---|
| Sistémico: cambio de régimen / OOD | Superficie fuera del soporte entrenado en choques de volatilidad | Parámetros explosivos, violaciones de no-arbitraje, malas coberturas | Circuit-breaker obligatorio + estrato de estrés en el entrenamiento + prueba OOD sintética |
| Idiosincrático: calidad de datos | Desfase entre calls y puts, strikes faltantes, spreads explosivos | Artefactos que el surrogate hereda | Filtros OTM, ponderación por spread, enmascarar nodos ausentes (operador $M$) |
| Ejecución: complejidad | Duplicación (pricer + PyTorch + exportador + C++), deriva de pesos | Parálisis del proyecto en depuración de build | Versiones fijadas, `.bin` versionado con metadatos de grilla, test de paridad en CI |
| Metodológico | Fuga temporal; comparar contra baselines débiles | Conclusiones infladas | Partición por fecha; baseline fuerte (E0) y baseline ingenuo declarados |
| Identificabilidad | Valles planos en κ, v̄, σ | Θ inestables día a día | Reportar estabilidad; no vender Θ como "parámetros físicos" |

### 8.3 Episodios de estrés candidatos (verificar fechas y disponibilidad de datos)

feb-2018, mar-2020, ago-2024, abr-2025. El reporte de Gemini añade un evento de **mar-2026** con cifras de VIX: `[NO VERIFICADO]`, no usarlo como hecho.

---

## 9. Guía para el agente de código

### 9.1 Reglas

1. Lee este documento completo antes de escribir código y trabaja **una fase a la vez**, en orden.
2. Test primero: cada módulo nace con su test (3.4, T1.4, T6.1).
3. Nunca declares una mejora de rendimiento sin un benchmark que la respalde en esta misma sesión.
4. No inventes referencias ni cifras. Si dudas de una cita, márcala `[VERIFICAR]` y sigue.
5. No subas datos de proveedores ni credenciales. Revisa `.gitignore` antes de cada commit.
6. Fija versiones. No agregues dependencias pesadas (LibTorch) sin justificarlo contra D3.
7. Cuando un gate no se cumpla, repórtalo con el valor medido en lugar de ajustar el umbral en silencio.
8. Commits pequeños, mensajes claros, una fase por rama.

### 9.2 Prompt de arranque (Fase 1)

```text
Lee NEURAL_HESTON_SPEC.md completo. Vamos a ejecutar solo la Fase 1 (pricer clásico).
1) Crea el esqueleto CMake mínimo (C++20, Eigen 5.0.x fijado, Catch2 o GoogleTest).
2) Implementa heston_cf.hpp (sección 3.2) y cos_pricer.hpp (3.3) con nodos SoA alineados.
3) Implementa las derivadas analíticas (3.4) y el Jacobiano de precios.
4) Escribe los tests T1.4: vs QuantLib (Python), convergencia en N, Jacobiano vs diferencias finitas.
5) Informa los errores máximos medidos contra el gate G1. No pases a la Fase 2.
Si alguna fórmula de la spec no coincide con QuantLib, el test manda: corrige la implementación y deja una nota.
```

---

## 10. Publicación en GitHub

### 10.1 Cuándo publicar

- **Después de G2:** repo público con el baseline clásico en C++ y el benchmark contra el baseline ingenuo. Ya tiene valor propio, genera historial de commits y no hace afirmaciones neuronales.
- **Después de G7–G8:** release `v0.1.0` con resultados completos.

### 10.2 Metadatos del repo

- **Nombre:** `neural-heston-calibration`
- **Descripción:** *Heston calibration in C++20/Python: analytic-gradient COS+LM baseline vs neural surrogate & warm-start, with an honest benchmark on SPX.*
- **Topics:** `quantitative-finance`, `heston-model`, `option-pricing`, `volatility-surface`, `calibration`, `cpp20`, `pytorch`, `nanobind`, `eigen`, `derivatives`.
- **Licencia:** MIT (o Apache-2.0 si quieres cláusula de patentes). Compatibilidad: Eigen es MPL-2.0 (solo cabeceras; aceptable), nanobind es BSD-3, QuantLib BSD (solo en tests). Verificar las licencias de PyTorch y de cualquier otra dependencia al fijar versiones.

### 10.3 CI

GitHub Actions: build + `ctest` + `pytest` en gcc y clang (Linux), `clang-format --dry-run`, `ruff`. **Los benchmarks no corren en CI** (hardware compartido y ruidoso): se ejecutan en tu máquina, con el hardware declarado, y los artefactos se suben a `results/`.

### 10.4 Estructura del README (en inglés)

1. **TL;DR** — 3 líneas: qué mide el proyecto y el hallazgo principal *una vez medido*.
2. **Why this project** — el problema del baseline débil.
3. **Method** — COS + analytic Jacobian, surrogate (Sobolev), inverse warm-start, circuit-breaker.
4. **Results** — tabla 7.5 con datos reales, hardware y flags.
5. **Limitations** — sección obligatoria: Heston vs SPX, identificabilidad, OOD, datos, un solo hardware.
6. **Reproduce** — `make repro-small` y `make repro` (con datos propios).
7. **References** y **Citation** (`CITATION.cff`).

### 10.5 Higiene antes de hacerlo público

- [ ] `git log -p` sin credenciales ni datos de proveedores.
- [ ] `data/` ignorado; solo el mini dataset sintético en el repo.
- [ ] Cada cita de la sección 12 con estado `[VERIFICADO]` o marcada.
- [ ] Ninguna cifra de rendimiento sin celda correspondiente en `results/`.
- [ ] Sección *Limitations* escrita.
- [ ] Versión fijada en `CITATION.cff`; opcional: DOI vía Zenodo en el primer release.

### 10.6 Cómo contarlo (solo después de tener resultados)

Plantilla para CV/LinkedIn, rellenando con **tus** números medidos:
*"Built a C++20/Python Heston calibration stack (COS + analytic Jacobian + LM). Benchmarked against a neural surrogate and inverse warm-start on SPX: [X] ms p99 classical vs [Y] ms neural, [Z] vol-point RMSE; documented where each approach fails."*

---

## 11. Auditoría del reporte de Gemini

Verificación del 2026-11-21. **Corregido** = el reporte original estaba equivocado o desactualizado.

| Afirmación del reporte | Estado | Acción |
|---|---|---|
| Cui, del Baño Rollin & Germano, EJOR 263(2):625–638 (2017): CF estable + gradiente analítico + LM | `[VERIFICADO]` | Usar |
| Gradiente analítico "~4–6× más rápido que diferencias finitas" | **Corregido:** el paper dice ≈10× | Medir el propio |
| Eigen 3.4.1 | **Corregido:** la serie vigente es 5.0.x (5.0.0, 30-sep-2025; 5.0.1) | Usar 5.0.x |
| nanobind 3.1.0 (sep-2026) | `[VERIFICADO]` (también 3.0.0 del 22-ago-2026) | Usar |
| QuantLib 1.44 (oct-2026) | `[VERIFICADO]` (13-oct-2026) | Usar solo en tests |
| Horvath–Muguruza–Tomas, Quant. Finance 21(1):11–27 (2021); repo *NN-StochVol-Calibrations* | `[VERIFICADO]` | Citar |
| Bayer–Stemper (arXiv:1810.03399); Bayer–Horvath–Muguruza–Stemper–Tomas (arXiv:1908.08806) | `[VERIFICADO]` | Citar |
| Baschetti–Bormetti–Rossi, Quant. Finance 24(9):1263–1285 (2024), arXiv:2306.11061 | `[VERIFICADO]` | Citar |
| Repo `fabioBaschetti/random-grid-NN-calib` | `[NO VERIFICADO]` | No enlazar sin comprobar |
| Zhang–Amici–Morandotti, "Calibrating the Heston model with deep differential networks", arXiv:2407.15536 (v2, ene-2025) | `[VERIFICADO]` | Citar. La "revisión may/sep-2026" del reporte: `[NO VERIFICADO]` |
| Sridi et al., arXiv:2309.07843, repo `asridi/DML-Calibration-Heston-Model` | `[VERIFICADO]` (lo encontré yo; el reporte no lo cita) | Candidato a referencia y a reproducción |
| Abi Jaber & Li, arXiv:2401.03345: los modelos rough no superan de forma consistente a sus equivalentes markovianos en SPX para plazos cortos | `[VERIFICADO]` (resultado cualitativo) | Citar |
| Cifras "0.50 vs 0.55 vol points" y "63 % de los días" | `[NO VERIFICADO]` | No citar |
| Rømer (2022), Hernández (2016/2017), Itkin (2019), Roeder–Dimitroff (2020), Huge–Savine (2020), Liu et al. (2019) | Existen en mi conocimiento; **no re-verificados** hoy salvo Liu–Borovykh–Grzelak–Oosterlee (arXiv:1904.10523, vía referencias cruzadas) | Verificar identificadores antes de publicar |
| Alpaca: "no cubre SPX" (lo dije en mi diagnóstico previo) | **Corregido:** Alpaca ya lista opciones SPX/SPXW para trading; los datos de mercado del índice no se ofrecen | Ver 5.2; confirmar si hay cotizaciones de opciones SPX |
| Precios de datos (IBKR 1.50, Databento 199, DataShop 86), dataset Kaggle/OptionsDX | `[NO VERIFICADO]` | Comprobar en la fuente |
| Cifras de VIX de 2026, evento de mar-2026, "16–18 parámetros" de Vola Dynamics, RMSE 1.99–2.87 de MDPI 2026, Feller 8.5×, SPY 31.9 % de días con dividendo negativo | `[NO VERIFICADO]` | **No citar** |
| Benchmark ORT 6.81 µs vs LibTorch 7.42 µs | `[NO VERIFICADO]` | No citar; medir si decides usar ORT |
| Todas las cifras de FLOPs y latencias | `[ESTIMADO]` por Gemini | Reemplazar por mediciones |
| Jacobiano forward-mode con 5 tangentes es mucho más eficiente que reverse para 5 entradas y 150–200 salidas | Correcto cualitativamente (razonamiento estándar de AD) | Implementar (3.8) |
| Tensión interna: "la red no se justifica por latencia" vs "surrogate de grilla 5–15× más rápido" | Inconsistencia del reporte | Resolver con H1/H2 |

---

## 12. Bibliografía

Entradas marcadas ✔ fueron confirmadas el 2026-11-21. Las demás: confirmar autores, año y revista antes de publicar.

- ✔ Cui, Y., del Baño Rollin, S., Germano, G. (2017). *Full and fast calibration of the Heston stochastic volatility model.* European Journal of Operational Research 263(2), 625–638. arXiv:1511.08718.
- ✔ Horvath, B., Muguruza, A., Tomas, M. (2021). *Deep learning volatility.* Quantitative Finance 21(1), 11–27. arXiv:1901.09647. Código: `NN-StochVol-Calibrations`.
- ✔ Bayer, C., Stemper, B. (2018). *Deep calibration of rough stochastic volatility models.* arXiv:1810.03399.
- ✔ Bayer, C., Horvath, B., Muguruza, A., Stemper, B., Tomas, M. (2019). *On deep calibration of (rough) stochastic volatility models.* arXiv:1908.08806.
- ✔ Baschetti, F., Bormetti, G., Rossi, P. (2024). *Deep calibration with random grids.* Quantitative Finance 24(9), 1263–1285. arXiv:2306.11061.
- ✔ Zhang, C., Amici, G., Morandotti, M. (2024/25). *Calibrating the Heston model with deep differential networks.* arXiv:2407.15536.
- ✔ Sridi, A. et al. (2023). *Applying deep learning to calibrate stochastic volatility models.* arXiv:2309.07843.
- ✔ Abi Jaber, E., Li, S. (2024/25). *Volatility models in practice: Rough, path-dependent or Markovian?* arXiv:2401.03345.
- ✔ Liu, S., Borovykh, A., Grzelak, L. A., Oosterlee, C. W. (2019). *A neural network-based framework for financial model calibration.* Journal of Mathematics in Industry 9. arXiv:1904.10523.
- Heston, S. L. (1993). *A closed-form solution for options with stochastic volatility…* Review of Financial Studies 6(2), 327–343.
- Albrecher, H., Mayer, P., Schoutens, W., Tistaert, J. (2007). *The little Heston trap.* Wilmott Magazine.
- Fang, F., Oosterlee, C. W. (2008/09). *A novel pricing method for European options based on Fourier-cosine series expansions.* SIAM Journal on Scientific Computing 31(2), 826–848.
- Carr, P., Madan, D. (1999). *Option valuation using the fast Fourier transform.* Journal of Computational Finance 2(4), 61–73.
- Junike, G., Pankrashkin, K. (2022). *Precise option pricing by the COS method — How to choose the truncation range.* (verificar revista).
- Jäckel, P. (2015). *Let's be rational.* Wilmott Magazine.
- Hernandez, A. (2016/17). *Model calibration with neural networks.* SSRN / Risk Magazine.
- Huge, B., Savine, A. (2020). *Differential machine learning.* (verificar arXiv y Risk).
- Itkin, A. (2019/20). *Deep learning calibration of option pricing models: some pitfalls and solutions.* (verificar identificador).
- Roeder, D., Dimitroff, G. (2020). *Volatility model calibration with neural networks: a comparison between direct and indirect methods.* (verificar identificador).
- Rømer, S. E. (2022). *Empirical analysis of rough and classical stochastic volatility models to the SPX and VIX markets.* Quantitative Finance. (verificar datos de publicación).
- Gatheral, J., Jacquier, A. (2014). *Arbitrage-free SVI volatility surfaces.* Quantitative Finance 14(1). (condición de mariposa; verificar).

---

## Apéndice A — Checklist de arranque (primer día)

- [ ] Crear el repo privado y subir este documento como `NEURAL_HESTON_SPEC.md`.
- [ ] Ejecutar el prompt de la sección 9.2 con el agente de código.
- [ ] Anotar en un `CHANGELOG.md` cada gate cumplido, con el valor medido.
- [ ] Decidir la fuente de datos SPX de la sección 5.2 antes de llegar a la Fase 3.
