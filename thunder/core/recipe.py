import thunder
from thunder import jit, Transform, Executor


class Lookaside:
    def __init__(self, fn, replace_with):
        self._fn = fn
        self._replace_with = replace_with


class ThunderRecipe:
    def __init__(self):
        pass

    def validate(self, model):
        # this is supposed to raise
        pass

    # def setup_operators(self) -> list[Operator]:
    #     # this is for registering custom kernels
    #     return None

    def setup_lookasides(self) -> list[Lookaside]:
        return None

    def setup_transforms(self) -> list[Transform]:
        return None

    # TODO: we are missing registering executors
    #       and operators
    def setup_executors(self) -> list[Executor]:
        return None

    def setup_config(self):
        return {}

    def setup(self, model):
        self.validate(model)

        lookasides = self.setup_lookasides()

        if lookasides is not None:
            for lookaside in lookasides:
                thunder.jit_ext.register_general_jit_lookaside(lookaside._fn)(
                    thunder.jit_ext.interpreter_needs_wrap()(
                        lookaside._replace_with
                    )
                )

        self.lookasides = lookasides
        self.executors = self.setup_executors()
        self.transforms = self.setup_transforms()


from typing import List, Sequence

def compile(model, recipe: None | ThunderRecipe | List[ThunderRecipe] = None):
    recipes = recipe if isinstance(recipe, Sequence) else [recipe]

    transforms = []
    executors = []
    config = {}

    for r in recipes:
        r.setup(model)
        transforms.extend(r.transforms)
        executors.extend(r.executors)
        config.update(r._config)

    jmodel = jit(model,
                 transforms=transforms,
                 executors=executors,
                 **recipe._config)

    return jmodel
