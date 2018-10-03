# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = []

import numpy as np

import theano
import theano.tensor as tt

from .kepler import Primary, Secondary, System


class TheanoPrimary(object):

    def __init__(self, lmax=0):
        self.lmax = 0


class StarryOp(tt.Op):

    def get_system(self, primary_lmax=):
        pass

    def __init__(self):
        # Star limb darkening
        itypes = [tt.dvector()]
        self.param_names = [["A.u_{1}", "A.u_{2}"]]

        # Planet hot spot
        itypes += [tt.vector()]
        self.param_names += [["b.Y_{1,-1}", "b.Y_{1,0}", "b.Y_{1,1}"]]

        # Planet orbital parameters
        itypes += [tt.dscalar() for i in range(9)]
        self.param_names += [[n] for n in ["b.r", "b.L", "b.a", "b.porb", "b.prot", "b.inc", "b.ecc", "b.w", "b.lambda0"]]

        # Times
        itypes += [tt.dvector()]
        self.param_names += [["time"]]

        self.itypes = tuple(itypes)

        # Output will always be a vector
        self.otypes = tuple([tt.dvector()])

        self._grad_op = StarryGradOp(self)

    def make_node(self, *args):
        args = [tt.as_tensor_variable(a) for a in args]
        return theano.Apply(self, args, [args[-1].type()])

    def infer_shape(self, node, shapes):
        """A required method that returns the shape of the output"""
        return shapes[-1],

    def perform(self, node, inputs, outputs):
        """A required method that actually executes the operation"""
        star, planet, system = build_system(*(inputs[:-1]))
        system.compute(np.array(inputs[-1]))
        outputs[0][0] = np.array(system.lightcurve)

    def grad(self, inputs, gradients):
        return self._grad_op(*(inputs + gradients))


class StarryGradOp(tt.Op):

    def __init__(self, base_op):
        self.base_op = base_op

        # This operation will take the original inputs and the gradient
        # seed as input
        types = list(self.base_op.itypes)
        self.nout = len(self.base_op.otypes)
        self.itypes = tuple(types + list(self.base_op.otypes))
        self.otypes = tuple(types)

    def make_node(self, *args):
        args = [tt.as_tensor_variable(a) for a in args]
        return theano.Apply(self, args, [a.type() for a in args[:-self.nout]])

    def infer_shape(self, node, shapes):
        return shapes[:-self.nout]

    def perform(self, node, inputs, outputs):
        star, planet, system = build_system(*(inputs[:-self.nout-1]))
        system.compute(np.array(inputs[-self.nout-1]), gradient=True)
        grads = system.gradient

        try:
            for i, param in enumerate(self.base_op.param_names):
                if len(param) > 1:
                    g = []
                    for j, name in enumerate(param):
                        g.append(np.sum(np.array(grads[name]) * np.array(inputs[-1])))
                    outputs[i][0] = np.squeeze(g)
                else:
                    if param[0] == "time":
                        outputs[i][0] = np.array(grads[param[0]]) * np.array(inputs[-1])
                    else:
                        outputs[i][0] = np.array(np.sum(np.array(grads[param[0]]) * np.array(inputs[-1])))
        except:
            print(grads.keys())
            raise
