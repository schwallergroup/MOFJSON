"""The structure task suite: algorithmically generated, automatically graded MOF questions.

Phase 12.
"""

from mofjson.tasks.build import BuildTaskSuite, FreezeSplit, Split, SuiteReport
from mofjson.tasks.contamination import (
    FAMOUS_MOFS,
    BuildControlArm,
    ClassifySource,
    ScanForLeaks,
)
from mofjson.tasks.generation import (
    CheckConstraints,
    Constraint,
    ConstraintSet,
    GenerateConstraintSets,
    GenerateGenerationItems,
)
from mofjson.tasks.grade import GradeRun, WilsonInterval
from mofjson.tasks.graders import GRADERS, Grade, GradeResult
from mofjson.tasks.item import ItemManifest, ReadItems, TaskItem, WriteItems
from mofjson.tasks.navigation import NAVIGATION_TASKS, GenerateNavigationItems
from mofjson.tasks.representations import REPRESENTATIONS, Parse, Render
from mofjson.tasks.sampling import BalanceByAnswer, StratifiedSample
from mofjson.tasks.translation import (
    ALL_PAIRS_MATRIX,
    TRANSLATION_PAIRS,
    GenerateTranslationItems,
)

__all__ = [
    "ALL_PAIRS_MATRIX",
    "BalanceByAnswer",
    "BuildControlArm",
    "BuildTaskSuite",
    "CheckConstraints",
    "ClassifySource",
    "Constraint",
    "ConstraintSet",
    "FAMOUS_MOFS",
    "FreezeSplit",
    "GRADERS",
    "GenerateConstraintSets",
    "GenerateGenerationItems",
    "GenerateNavigationItems",
    "GenerateTranslationItems",
    "Grade",
    "GradeResult",
    "GradeRun",
    "ItemManifest",
    "NAVIGATION_TASKS",
    "Parse",
    "REPRESENTATIONS",
    "ReadItems",
    "Render",
    "ScanForLeaks",
    "Split",
    "StratifiedSample",
    "SuiteReport",
    "TRANSLATION_PAIRS",
    "TaskItem",
    "WilsonInterval",
    "WriteItems",
]
