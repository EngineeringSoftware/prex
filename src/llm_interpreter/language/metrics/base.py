import math
from abc import ABC, abstractmethod
from enum import StrEnum
from dataclasses import dataclass, field
from collections import defaultdict, deque


class HalsteadMetric(ABC):
    class Measures(StrEnum):
        DISTINCT_OPERATORS = "eta1"
        DISTINCT_OPERANDS = "eta2"
        TOTAL_OPERATORS = "N1"
        TOTAL_OPERANDS = "N2"
        VOCABULARY = "Vocabulary"
        LENGTH = "Length"
        BIT_LENGTH = "Bitlength"
        VOLUME = "Volume"
        DIFFICULTY = "Difficulty"
        EFFORT = "Effort"
        TIME = "Time"

    @abstractmethod
    def get_halstead_metrics(self, program: str) -> dict:
        pass

    # fed

    @classmethod
    def compute_halstead_metrics(
        cls,
        num_distinct_operators: int,
        num_distinct_operands: int,
        total_operators: int,
        total_operands: int,
    ) -> dict:
        vocabulary: int = num_distinct_operators + num_distinct_operands
        length: int = total_operators + total_operands
        bit_length: float = round(
            num_distinct_operators * math.log2(num_distinct_operators)
            + num_distinct_operands * math.log2(num_distinct_operands),
            2,
        )
        volume: float = round(length * math.log2(vocabulary), 2)
        difficulty: float = round(
            (num_distinct_operators / 2) * (total_operands / num_distinct_operands), 2
        )
        effort: float = round(difficulty * volume, 2)
        time: float = round(effort / 18, 2)

        return {
            HalsteadMetric.Measures.DISTINCT_OPERATORS.value: num_distinct_operators,
            HalsteadMetric.Measures.DISTINCT_OPERANDS.value: num_distinct_operands,
            HalsteadMetric.Measures.TOTAL_OPERATORS.value: total_operators,
            HalsteadMetric.Measures.TOTAL_OPERANDS.value: total_operands,
            HalsteadMetric.Measures.VOCABULARY.value: vocabulary,
            HalsteadMetric.Measures.LENGTH.value: length,
            HalsteadMetric.Measures.BIT_LENGTH.value: bit_length,
            HalsteadMetric.Measures.VOLUME.value: volume,
            HalsteadMetric.Measures.DIFFICULTY.value: difficulty,
            HalsteadMetric.Measures.EFFORT.value: effort,
            HalsteadMetric.Measures.TIME.value: time,
        }

    # fed


# ssalc


class ExtendedCyclomaticMetric(ABC):
    class Measures(StrEnum):
        NUM_IF = "NumIf"
        NUM_WHILE = "NumWhile"
        NUM_AND = "NumAnd"
        NUM_OR = "NumOr"
        MAX_BLOCK_DEPTH = "MaxDepth"
        MAX_NESTED_LOOP = "MaxNestedLoop"
        MAX_NESTED_IF = "MaxNestedIf"
        CC = "CC"

    @abstractmethod
    def get_extended_cyclomatic_metrics(self, program: str) -> dict:
        pass

    # fed

    @classmethod
    def compute_extended_cyclomatic_metrics(
            cls, num_if: int, num_while: int, num_and: int, num_or: int, num_max_depth: int, num_max_nest_loop: int, num_max_nest_if: int
    ) -> dict:
        cc = num_if + num_while + num_and + num_or + 1
        return {
            ExtendedCyclomaticMetric.Measures.NUM_IF.value: num_if,
            ExtendedCyclomaticMetric.Measures.NUM_WHILE.value: num_while,
            ExtendedCyclomaticMetric.Measures.NUM_AND.value: num_and,
            ExtendedCyclomaticMetric.Measures.NUM_OR.value: num_or,
            ExtendedCyclomaticMetric.Measures.MAX_BLOCK_DEPTH.value: num_max_depth,
            ExtendedCyclomaticMetric.Measures.MAX_NESTED_LOOP.value: num_max_nest_loop,
            ExtendedCyclomaticMetric.Measures.MAX_NESTED_IF.value: num_max_nest_if,
            ExtendedCyclomaticMetric.Measures.CC.value: cc,
        }

    # fed


# ssalc


class DepDegreeMetric(ABC):
    class Measures(StrEnum):
        DEP_DEGREE = "DepDegree"
        DEP_DEGREE_PER_VARIABLE = "DepDegreePerVariable"

    @dataclass
    class Node:
        name: str
        defs: set[str]
        uses: set[str]
        succs: list[int] = field(default_factory=list)

    # ssalc

    @abstractmethod
    def get_depdegree_metrics(self, program: str) -> dict:
        pass

    # fed

    @classmethod
    def compute_reaching_definitions(cls, nodes: list) -> list:
        defs_of_var: dict = {}
        for i, n in enumerate(nodes):
            for v in n.defs:
                defs_of_var.setdefault(v, set()).add(i)
            # rof
        # rof

        IN = [dict() for _ in nodes]
        OUT = [dict() for _ in nodes]

        def dict_union(a: dict, b: dict) -> dict:
            out = {k: set(v) for k, v in a.items()}
            for k, s in b.items():
                out.setdefault(k, set()).update(s)
            # rof
            return out

        # fed

        changed = True
        while changed:
            changed = False
            for i, n in enumerate(nodes):
                preds = [p for p, P in enumerate(nodes) if i in P.succs]
                newIN: dict = {}
                for p in preds:
                    newIN = dict_union(newIN, OUT[p])
                # rof
                newOUT: dict = {k: set(v) for k, v in newIN.items()}
                for v in n.defs:
                    newOUT[v] = set()
                # rof
                for v in n.defs:
                    newOUT.setdefault(v, set()).add(i)
                # rof

                if newIN != IN[i] or newOUT != OUT[i]:
                    IN[i], OUT[i] = newIN, newOUT
                    changed = True
                # fi
            # rof
        # elihw
        return IN

    # fed

    @classmethod
    def compute_depdegree(
        cls,
        nodes: list,
    ) -> dict:
        RD_in: list = cls.compute_reaching_definitions(nodes)
        per_var: dict = defaultdict(int)
        total: int = 0
        for i, n in enumerate(nodes):
            for v in n.uses:
                reaching = RD_in[i].get(v, set())
                per_var[v] += len(reaching)
                total += len(reaching)
            # rof
        # rof
        return {
            DepDegreeMetric.Measures.DEP_DEGREE.value: total,
            DepDegreeMetric.Measures.DEP_DEGREE_PER_VARIABLE.value: per_var,
        }

    # fed


# ssalc
