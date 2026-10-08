import time
from typing import Callable, Optional, Type, Union

import lightweaver as lw
from lightweaver.deprecation import accepts_old_kwargs
import numpy as np


class PromModel:
    """
    Base Class for PROM-style models.

    This exists to provide the `iterate_se` method to the derived models, as
    such, your `__init__` needs to provide `ctx`: a `lw.Context` describing
    the model to compute.
    """

    def __init__(self, ctx: lw.Context):
        self.ctx = ctx

    def compute_rays(
        self,
        wavelengths: Optional[np.ndarray] = None,
        mus: Optional[Union[float, np.ndarray, dict]] = None,
        compute_rays_kwargs: Optional[dict] = None,
    ):
        """
        Compute the formal solution through a converged simulation for a
        particular ray (or set of rays). The wavelength range can be adjusted
        to focus on particular lines.
        Parameters
        ----------
        wavelengths : np.ndarray, optional
            The wavelengths at which to compute the solution (default: None,
            i.e. the original grid).
        mus : float or sequence of float or dict, optional
            The cosines of the angles between the rays and the z-axis to use,
            if a float or sequence of float then these are taken as muz. If a
            dict, then it is expected to be dictionary unpackable
            (double-splat) into atmos.rays, and can then be used for
            multi-dimensional atmospheres.
        compute_rays_kwargs :  dict, optional
            Any extra kwargs to be passed to `ctx.compute_rays`.
        Returns
        -------
        intensity : np.ndarray
            The outgoing intensity for the chosen rays.
        """
        if compute_rays_kwargs is None:
            compute_rays_kwargs = {}

        if wavelengths is None:
            wavelengths = self.ctx.spect.wavelength

        self.ctx.atmos.py_atmos.final_synthesis = True
        I = self.ctx.compute_rays(
            wavelengths=wavelengths, mus=mus, **compute_rays_kwargs
        )
        self.ctx.atmos.py_atmos.final_synthesis = False
        return I

    @accepts_old_kwargs
    def iterate_se(
        self,
        Nscatter: int = 3,
        max_iter: int = 2000,
        prd: bool = False,
        J_tol: float = 5e-3,
        pops_tol: float = 1e-3,
        rho_tol: Optional[float] = None,
        prd_iter_tol: float = 1e-2,
        max_prd_sub_iter: int = 3,
        print_interval: float = 0.2,
        quiet: bool = False,
        convergence: Optional[Type[lw.ConvergenceCriteria]] = None,
        return_final_convergence: bool = False,
        update_model: Optional[Callable[..., None]] = None,
        update_model_kwargs: Optional[dict] = None,
    ):
        r"""
        Iterate the context towards statistical equilibrium solution. Slightly
        modified variant of the core `lw.iterate_ctx_se` to add and handle the
        `update_model` callback.

        Parameters
        ----------
        Nscatter : int, optional
            The number of lambda iterations to perform for an initial estimate of J
            (default: 3).
        max_iter : int, optional
            The maximum number of iterations (including Nscatter) to take (default:
            2000).
        prd: bool, optional
            Whether to perform PRD subiterations to estimate rho for PRD lines
            (default: False).
        J_tol: float, optional
            The maximum relative change in J from one iteration to the next
            (default: 5e-3).
        pops_tol: float, optional
            The maximum relative change in an atomic population from one iteration
            to the next (default: 1e-3).
        rho_tol: float, optional
            The maximum relative change in rho for a PRD line on the final
            subiteration from one iteration to the next. If None, the change in rho
            will not be considered in judging convergence (default: None).
        prd_iter_tol: float, optional
            The maximum relative change in rho for a PRD line below which PRD
            subiterations will cease for this iteration (default: 1e-2).
        max_prd_sub_iter : int, optional
            The maximum number of PRD subiterations to make, whether or not rho has
            reached the tolerance of prd_iter_tol (which isn't necessary every
            iteration). (Default: 3)
        print_interval : float, optional
            The interval between printing the update size information in seconds. A
            value of 0.0 will print every iteration (default: 0.2).
        quiet : bool, optional
            Overrides any other print arguments and iterates silently if True.
            (Default: False).
        convergence : derived ConvergenceCriteria class, optional
            The ConvergenceCriteria version to be used in determining convergence.
            Will be instantiated by this function, and the `is_converged` method
            will then be used.  (Default: DefaultConvergenceCriteria).
        return_final_convergence : bool, optional
            Whether to return the IterationUpdate objects used in the final
            convergence decision, if True, these will be returned in a list as the
            second return value. (Default: False).
        update_model : callable(PromModel, bool, \*\*kwargs), optional
            A function to use to update model parameters based on iteration
            (e.g. scale populations to maintain pressure), receives the model,
            and whether it should print based on the current iteration progress
            loop.
        update_model_kwargs : dict, optional
            Extra kwargs to be passed to `update_model`.

        Returns
        -------
        it : int
            The number of iterations taken.
        final_iteration_updates : List[IterationUpdate], optional
            The final IterationUpdates computed, if requested by `return_final_convergence`.
        """
        prev_print = 0.0
        print_now = True
        always_print = print_interval == 0.0
        start_time = time.time()
        ctx = self.ctx

        if update_model_kwargs is None:
            update_model_kwargs = {}

        if convergence is None:
            convergence = lw.DefaultConvergenceCriteria
        conv = convergence(ctx, J_tol, pops_tol, rho_tol)

        for it in range(max_iter):
            J_update: lw.IterationUpdate = ctx.formal_sol_gamma_matrices()
            if not quiet and (
                always_print or ((now := time.time()) >= prev_print + print_interval)
            ):
                print_now = True
                if not always_print:
                    prev_print = now

            if not quiet and print_now:
                print(f"-- Iteration {it}:")
                print(J_update.compact_representation())

            if it < Nscatter:
                if not quiet and print_now:
                    print("    (Lambda iterating background)")
                # NOTE(cmo): reset print state
                print_now = False
                continue

            pops_update: lw.IterationUpdate = ctx.stat_equil()
            if not quiet and print_now:
                print(pops_update.compact_representation())

            drho_update: Optional[lw.IterationUpdate]
            if prd:
                drho_update = ctx.prd_redistribute(max_iter=max_prd_sub_iter, tol=prd_iter_tol)
                if not quiet and print_now and drho_update is not None:
                    print(drho_update.compact_representation())
            else:
                drho_update = None

            terminate = conv.is_converged(J_update, pops_update, drho_update)

            if it > Nscatter + 1 and terminate:
                if not quiet:
                    end_time = time.time()
                    duration = end_time - start_time
                    line = "-" * 80
                    if print_now:
                        print("Final Iteration shown above.")
                    else:
                        print(line)
                        print(f"Final Iteration: {it}")
                        print(line)
                        print(J_update.compact_representation())
                        print(pops_update.compact_representation())
                        if prd and drho_update is not None:
                            print(drho_update.compact_representation())
                    print(line)
                    print(
                        f"Context converged to statistical equilibrium in {it}"
                        f" iterations after {duration:.2f} s."
                    )
                    print(line)
                if return_final_convergence:
                    final_convergence = [J_update, pops_update]
                    if prd and drho_update is not None:
                        final_convergence.append(drho_update)
                    return it, final_convergence
                else:
                    return it

            if update_model is not None:
                update_model(self, print_now, **update_model_kwargs)

            # NOTE(cmo): reset print state
            print_now = False
        else:
            if not quiet:
                line = "-" * 80
                end_time = time.time()
                duration = end_time - start_time
                print(line)
                print(f"Final Iteration: {it}")
                print(line)
                print(J_update.compact_representation())
                print(pops_update.compact_representation())
                if prd and drho_update is not None:
                    print(drho_update.compact_representation())
                print(line)
                print(
                    f"Context FAILED to converge to statistical equilibrium after {it}"
                    f" iterations (took {duration:.2f} s)."
                )
                print(line)
            if return_final_convergence:
                final_convergence = [J_update, pops_update]
                if prd and drho_update is not None:
                    final_convergence.append(drho_update)
                return it, final_convergence
            else:
                return it
