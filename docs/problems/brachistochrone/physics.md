# Physics

```{variable-context} gemseo_tempo.problems.brachistochrone.model:REGISTRY
```

A bead slides down a wire from the origin to a target point. The shape of
the wire is the answer; the angle it makes with the horizontal is what the
optimizer chooses.

**Assumptions.**
1. A point mass on a wire whose shape the path angle traces; no climbing,
   γ ∈ [0, π/2].
2. Uniform gravity {var}`g <symbol>` along +y.
3. Coulomb friction on the normal force the wire actually exerts, turning
   included, {eq}`brachistochrone-normal`; the classical problem drops the
   turning term {footcite:p}`ashby_brachistochrone_1975`, and the friction
   variant's reference is numerical, for this model.
4. Quadratic drag, with the coefficient of the bead's own shape,
   {eq}`brachistochrone-drag`.
5. The bead starts at rest at the origin.

## Kinematics

```{eval-rst}
.. plot:: _ext/gallery/brachistochrone_kinematics.py
   :include-source: false
```

The state is a position and a speed, the control is the path angle, and
with {var}`y <symbol>` measured downward the geometry is one line:

```{math}
:label: brachistochrone-kinematics
\dot{x} = v \cos\gamma, \qquad \dot{y} = v \sin\gamma
```

```{variables} gemseo_tempo.problems.brachistochrone.model:REGISTRY
:filter: tags=state,control
:columns: name, full_name, unit, bounds
```

## The forces

Two forces act beyond gravity, and each is written from the object rather
than from a convenient constant.

### Friction, from the normal force

Coulomb friction is proportional to the force the wire actually exerts,
which is *not* simply the normal component of the bead's weight. A bead on
a curved path is also being turned, and turning costs a centripetal force
the wire has to supply on top of holding the bead up. Newton's law
perpendicular to the path gives

```{math}
:label: brachistochrone-normal
\frac{N}{m} = \max\!\left(g \cos\gamma - v\, \dot{\gamma},\; 0\right)
```

The second term is the one usually dropped, and it is not small here: on
the converged friction variant the normal force reaches 17.6 m/s², against
9.3 for the weight component alone, and the descent takes 8.7% longer than
the same bead with the term left out.

The floor is not cosmetic. A wire can push a bead but not pull it, so the
normal force cannot go negative; left unfloored, a hard enough turn drives
it through zero, friction changes sign, and the model says that rubbing
against the wire makes the bead faster — which an optimizer will find. It
is smoothed rather than clipped, so the floor is differentiable where it
bites.

Reading {math}`\dot\gamma` costs something structural: the model has to ask
its transcription for the control's time-derivative, which single shooting
and collocation supply and indirect shooting cannot, since the control then
enters the Hamiltonian through its own rate and stationarity stops being a
pointwise condition. See {doc}`../../concepts/transcriptions`.

### Drag, from a cylinder

The bead is a circular cylinder in cross-flow, so its drag coefficient is
the textbook subcritical value for that shape rather than a number chosen
to make the figure look right. Writing the mass as the cylinder's own
volume times its density, the length cancels and the whole group collapses
to a function of one length scale:

```{math}
:label: brachistochrone-drag
\frac{k}{m}
  = \frac{\tfrac12 \rho_{air} C_d\, (d L)}{\rho_{bead} \tfrac{\pi}{4} d^2 L}
  = \frac{2 C_d \rho_{air}}{\pi \rho_{bead} d}
```

At {math}`C_d = 1.2`, a 20 mm bead of dense plastic in sea-level air gives
{math}`k/m = 0.049\ \mathrm{m^{-1}}` — which is, to within 2%, the number
this problem used to carry as a bare constant. The value was right; what
was missing was any reason for it.

## Dynamics

Gravity along the path, friction against it, and drag against it:

```{math}
:label: brachistochrone-dynamics
\dot{v} = g \sin\gamma - \mu \frac{N}{m} - \frac{k}{m} v^2
```

The three variants differ only in which forces are switched on: the ideal
bead has {math}`\mu = 0` and no air, the friction variant rubs, and the
drag variant flies. The constants, and what is uncertain about them:

```{variables} gemseo_tempo.problems.brachistochrone.model:REGISTRY
:filter: tags=constant
:columns: name, full_name, unit, distribution
```

The uncertainty sits on the drag *coefficient*, where a cylinder's really
is uncertain — about 1.0 to 1.3 with Reynolds number and surface finish —
rather than on the lumped group, which is a geometric fact once the shape
is fixed.

## Along the wire

Integrated in time, the model above is the right description of a vehicle
steering itself and the wrong one of a wire. Once the wire is built its
shape is fixed, and a bead meeting more friction than it was designed for
runs along the same wire, only slower; freezing an angle schedule in time
and changing the friction would instead bend the bead onto another path.
So when the constants are uncertain, on {doc}`uncertainty`, the wire is
described by its angle along its own length and the arc length *s*
replaces the time. With ζ = √s, which keeps the release from rest regular,

```{math}
:label: eq-brach-wire

\frac{dx}{d\zeta} = 2\zeta\cos\gamma, \quad
\frac{dy}{d\zeta} = 2\zeta\sin\gamma, \quad
\frac{dv}{d\zeta} = \frac{2\zeta}{v}\Big(g\sin\gamma - \mu\frac{N}{m}
  - \frac{k}{m}\,v^2\Big), \quad
\frac{dt}{d\zeta} = \frac{2\zeta}{v} .
```

The forces are the ones above. The normal force is still
{eq}`brachistochrone-normal`, with the wire turning the bead as fast as the
bead runs along its curvature, γ̇ = v dγ/ds = v (dγ/dζ) / 2ζ; the turning
term v² (dγ/dζ) / 2ζ then vanishes at the release, where the speed grows
like ζ. So does 2ζ/v stay finite there, tending to √(2/a) with *a* the
initial acceleration, and both limits are written out, so the integrator
never divides zero by zero. The time taken is then a state, read at the
end of the wire, and the end point is pure geometry: however the physics
changes, the bead arrives where the wire goes.

The wire's angle is a cubic spline through its nodes rather than a
piecewise-linear one, because the normal force reads its rate. A
piecewise-linear angle has a curvature that jumps from one interval to
the next, and friction that grows with the turn rate then pays the
optimizer to alternate steep and flat intervals: it settles on a sawtooth
and stalls short of the end point. The spline's curvature is continuous,
the optimum is the smooth wire, and it agrees with the time model: the
wire designed for the friction variant reaches that variant's reference
descent time to a part in ten thousand. See {mod}`~gemseo_tempo.problems.brachistochrone.wire`.

## References

```{footbibliography}
```
