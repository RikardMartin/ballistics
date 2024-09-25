# SimBal ballistics calculator
This is a tool for simulating trajectories of projectiles. It was created specifically for firearm bullets.
This tool was made just for fun, and there are for sure several possible improvements than can be made both to the application and to the algorithm.

## Requirements
You will need python > 3.n with the following libraries installed:
* numpy
* pandas
* matplotlib
* plotly
* streamlit

Start by opening a terminal and typing `streamlit run simbal.py`.


## Explanation of parameters
### Environmental constants
* rho = density of air
* g = acceleration from gravity
* Cd = drag coefficient

### Gear specifications
* L = characteristic length
* mu = flow speed
* m_gr = mass of projectile in grains (not grams)
* v0 = Initial speed of projectile

### Initial conditions
* theta_0_deg = Initial angle of projectile in degrees
* x0 = Initial horizontal position of projectile
* y0 = Initial vertical position of projectile

### Calculated constants
* A = bullet cross sectional area
* C = drag parameter
* theta_0 = Initial angle of projectile in radians
* vx0 = initial horizontal speed of projectile
* yx0 = initial vertical speed of projectile
* m = mass of projectile in kg


## Dynamics calculations

Denote acceleration $\vec{a} = (a_x, a_y)$, velocity $\vec{v} = (v_x, v_y)$ and position $\vec{r} = (r_x, r_y)$
where $\vec{a} = \frac{d\vec{v}}{dt}$ and  $\vec{v} = \frac{d\vec{r}}{dt}.$

Newtons force equation
$m\vec{a} = \vec{F} + m\vec{g}$
with standard drag force
$\vec{F} = -C\hat{v}v²$
gives us
$$\hat{x}: ma_x = -C\cos{(\theta(\vec{v}))}v² \qquad  \hat{y}: ma_y = -C\sin{(\theta(\vec{v}))}v²-mg$$

$$\Leftrightarrow$$

$$\hat{x}: a_x = -\frac{1}{m}C\cos{(\theta(\vec{v}))}v² \qquad  \hat{y}: a_y = -\frac{1}{m}(C\sin{(\theta(\vec{v}))}v²+mg)$$

Substituting
$\cos{(\theta(\vec{v}))} = v_x/v \quad \textrm{and} \quad \sin{(\theta(\vec{v}))} = v_y/v$
we can express acceleration as

$$\hat{x}: a_x = -\frac{1}{m}C v_x v \qquad  \hat{y}: a_y = -\frac{1}{m}(C v_y v + mg).$$

$\vec{a}$, $\vec{v}$ and $\vec{r}$ are related by

$$\vec{v} = \vec{v_0} + \vec{a}t$$

$$\vec{r} = \vec{r_0} + \vec{v_0}t + \frac{1}{2}\vec{a}t²$$

and

$$v = \sqrt{v_x² + v_y²}.$$


## Algorithm
Splitting the above equations into components and discretizing time gives us
$$\begin{align*}
    a_x(t_0) &= -\frac{C}{m}v_x(t_0)v(t_0) \\
    v_x(t_1) &= v_x(t_0) + a_x(t_0)\Delta t \\
    r_x(t_1) &= r_x(t_0) + v_x(t_0)\Delta t + \frac{1}{2}a_x(t_0)\Delta t² \end{align*}$$
for *x*,
$$\begin{align*}
    a_y(t_0) &= -\frac{1}{m}(Cv_y(t_0)v(t_0) + mg) \\
    v_y(t_1) &= v_y(t_0) + a_y(t_0)\Delta t \\
    r_y(t_1) &= r_y(t_0) + v_y(t_0)\Delta t + \frac{1}{2}a_y(t_0)\Delta t² \end{align*}$$
for *y* and
$$\begin{align*}
    v_1 &= \sqrt{v_x(t_1)² + v_y(t_1)²}
    \end{align*}$$
for *v*.

Iterating one timestep we get
$$\begin{align*}
    a_x(t_1) &= -\frac{C}{m}v_x(t_1)v(t_1) \\
    v_x(t_2) &= v_x(t_1) + a_x(t_1) \Delta t \\
    r_x(t_2) &= r_x(t_1) + v_x(t_1) \Delta t + \frac{1}{2}a_x(t_1) \Delta t²
    \end{align*}$$
and same pattern as above for *y*. We can generalize this as
$$\begin{align*}
    a_x(t-1) &= -\frac{C}{m}v_x(t-1)v(t-1) \\
    v_x(t) &= v_x(t-1) + a_x(t-1)\Delta t \\
    r_x(t) &= r_x(t-1) + v_x(t-1)\Delta t + \frac{1}{2}a_x(t-1)\Delta t²
    \end{align*}$$
and again similar for *y*. The equation for *v* becomes
$$\begin{align*}
    v(t) &= \sqrt{v_x(t)² + v_y(t)²}.
    \end{align*}$$
This defines our algorithm.


## Pseudocode
```bash
Initialize timesteps t with spacing dt.
Initialize starting state for a, v and r in vectors.

for timesteps t do:

    let v = sqrt(v_x(t)² + v_y(t)²)

    let a_x(t) = -C/m * v_x(t) * v
    let a_y(t) = -C/m * v_y(t) * v - g

    let v_x(t+1) = v_x(t) + a_x(t) * dt
    let v_y(t+1) = v_y(t) + a_y(t) * dt

    let r_x(t+1) = r_x(t) + v_x(t) * dt + 1/2 * a_x(t) * dt²
    let r_y(t+1) = r_y(t) + v_y(t) * dt + 1/2 * a_y(t) * dt²

```