from thunder import ThunderRecipe, Lookaside

import torch
import transformers


class CompileHFBert(ThunderRecipe):
    def __init__(self):
        super().__init__()

    def validate(self, model):
        if not isinstance(model, transformers.BertForSequenceClassification):
            raise ValueError("The model must be a BertForSequenceClassification")

    def setup_lookasides(self):
        warn_lookaside = Lookaside(
            fn=transformers.modeling_utils.PreTrainedModel.warn_if_padding_and_no_attention_mask,
            replace_with=lambda *args: None
        )

        if hasattr(torch, "compiler") and hasattr(torch.compiler, "is_compiling"):
            is_compiling = torch.compiler.is_compiling
        else:
            is_compiling = torch._dynamo.is_compiling

        is_compiling_lookaside = Lookaside(
            fn=is_compiling,
            replace_with=lambda *args: True
        )

        return [warn_lookaside, is_compiling_lookaside]

    def setup_transforms(self):
        return None

    def setup_executors(self):
        return None


class Quantize4Bit(ThunderRecipe):
    def __init__(self):
        super().__init__()

    def setup_transforms(self):
        from thunder.transforms.quantization import BitsAndBytesLinearQuant4bit
        return [BitsAndBytesLinearQuant4bit()]

    def setup_executors(self):
        from thunder.transforms.quantization import get_bitsandbytes_executor
        return [get_bitsandbytes_executor()]


def example_single():
    bert = transformers.BertForSequenceClassification(transformers.BertConfig())

    # Option 1
    recipe = CompileHFBert()
    t_bert = thunder.compile(bert, recipe)

    # Option 2
    recipe = CompileHFBert()
    t_bert = thunder.apply(recipe, bert)

    # Option 3
    # downside: we can't apply more than one recipe
    recipe = CompileHFBert()
    t_bert = recipe.apply(bert)

    # Option 4
    # downside: we can't apply more than one recipe
    recipe = CompileHFBert()
    t_bert = recipe.compile(bert)

    # Option 5
    # we usually don't do things in the constructor
    # constructor configures, method does
    t_bert = CompileHFBert(bert)

    t_bert.eval()
    x = torch.randint(1, 20, (1, 32))
    emb = t_bert(x)


def example_composition():
    bert = transformers.BertForSequenceClassification(transformers.BertConfig())

    # Option 1
    bert_recipe = CompileHFBert()
    quant_recipe = Quantize4Bit()
    t_bert = thunder.compile(bert, [bert_recipe, quant_recipe])

    # Option 2
    bert_recipe = CompileHFBert()
    quant_recipe = Quantize4Bit()
    t_bert = thunder.apply([bert_recipe, quant_recipe], bert)

    # Option 3
    # Ugly
    recipe = CompileHFBert(Quantize4Bit())
    t_bert = recipe.apply(bert)

    # Option 4
    # Ugly
    recipe = CompileHFBert(Quantize4Bit())
    t_bert = recipe.compile(bert)

    # Option 5
    # Ugly
    recipe = CompileHFBert()
    recipe = recipe.add(Quantize4Bit())
    t_bert = recipe.compile(bert)

    # Option 6
    # we usually don't do things in the constructor
    # constructor configures, method does
    t_bert = CompileHFBert(bert)

    # Option 7
    bert = CompileHFBert(bert)
    bert = Quantize4Bit(bert)
    # here bert is not a model, it's a callable
    # confusing
    # it could still be a model but we could 
    # stick extra properties to it, until we call jit
    # the main issue with this is that one needs
    # to think about the order
    # recipes could indeed have the concept of order
    # like traits, so the order becomes not important

    t_bert.eval()
    x = torch.randint(1, 20, (1, 32))
    emb = t_bert(x)


# here's an example of composition that is more manual
# before we venture in automating composition, we should
# probably take this route
# this way recipes become objects that are configurable
class QuantizedHFBert(ThunderRecipe):
    def __init__(self, quantize=False):
        super().__init__()
        self.bert_recipe = CompileHFBert()
        if quantize:
            self.quant_recipe = Quantize4Bit()
        self.quantize = quantize

    def setup_lookasides(self):
        return self.bert_recipe.setup_lookasides()

    def setup_transforms(self):
        if self.quantize:
            return self.quant_recipe.setup_transforms()
        return None
    
    def setup_executors(self):
        if self.quantize:
            return self.quant_recipe.setup_executors()
        return None
 
# what if I just want to run with the defaults?

def example_default():
    bert = transformers.BertForSequenceClassification(transformers.BertConfig())

    # Option 1
    t_bert = thunder.compile(bert)

    # Option 2
    # way too verbose
    recipe = thunder.BaseRecipe()
    t_bert = recipe.apply(bert)

    # ok, so a recipe is something optional
    
    # this should work
    t_bert = thunder.compile(bert)

    # this should work as well
    t_bert = thunder.compile(bert, recipe=CompileHFBert())

    # at this point, this should also work, for the power users
    # in this case we could have traits, or just follow the order
    t_bert = thunder.compile(bert, recipe=[CompileHFBert(), Quantize4Bit()])

    # but typically you want to have an actual recipe
    t_bert = thunder.compile(bert, recipe=QuantizedHFBert(quantize=True))

    # compile could also have lookasides, transforms and executors
    # that are not part of the recipe

    # when we apply multiple recipes, we can't actually call apply
    # because it calls jit