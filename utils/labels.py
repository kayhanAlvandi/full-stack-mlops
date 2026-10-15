"""Ground-truth label resolution for (plate, well) samples.

A label is a property of the physical sample ``(plate, well)`` -- not of a
directory, a model, or a training run -- so resolution takes a well list and
returns a mapping, nothing more. Discovering *which* wells exist is a separate
concern (see ``src/sample_selection.scan_directory``), which keeps this module
free of filesystem access and lets every caller scan the disk only once.

Lives in ``utils/`` (not ``src/``) because it's shared across every image that
needs labels: training, the standalone dataset builder, and
``monitoring/backfill_labels.py`` (lightweight monitoring image). It therefore
stays torch-free (stdlib + pandas), unlike ``src/dataset.py``, which needs
torch/cv2/PIL for the actual image-loading ``Dataset`` classes.

The label *provider* classes below (``MongoDBLabelsProvider``,
``CSVLabelsProvider``, ``DummyLabelsProvider``) are the single place a label
source is implemented -- there is no separate string/enum dispatcher. Every
caller that needs labels (``src.datamodule.MultiChannelDataModule``,
``src.sample_selection.DatasetSpec``/``scripts/build_dataset.py``) takes a
Hydra-style config (``{_target_: "utils.labels.XxxLabelsProvider", ...}``) and
instantiates it with ``hydra.utils.instantiate()``. Adding a new label source
is a new provider class here plus a YAML entry -- no changes to any caller.

The MongoDB connection string is hardcoded inside the external ``tools.loading``
module (shared team server); ``tools`` is imported lazily so that callers which
only need dummy/csv labels -- or only the ``limit_wells_per_label`` helper --
do not require the package to be installed.
"""
from __future__ import annotations

Well = tuple[str, str]

DEFAULT_DUMMY_CLASSES: tuple[str, ...] = ("ClassA", "ClassB", "ClassC", "ClassD")


def resolve_labels_from_mongodb(wells: list[Well], collection: str = "tags") -> dict[Well, str]:
    """Look up treatments for (plate, well) pairs in MongoDB.

    Sends a ``Plate``/``Well`` DataFrame through ``tools.loading.getCategories``
    and reads the resulting ``Treatment`` column.

    Only wells that resolved to a usable treatment are returned: a missing
    ``Treatment`` column yields ``{}`` (with a warning), and ``None``/``""``/NaN
    treatments are dropped. Letting those through would put a non-string into
    the label mapping, which later breaks ``sorted()`` in ``LabelEncoder.fit``
    with a confusing ``TypeError`` far from the real cause (bad upstream data).
    """
    import pandas as pd
    from tools.loading import getCategories

    if not wells:
        return {}

    df = pd.DataFrame({"Plate": [w[0] for w in wells], "Well": [w[1] for w in wells]})
    df.drop_duplicates(inplace=True)

    df_labels = getCategories(df, collection=collection)
    if "Treatment" not in df_labels.columns:
        print("WARNING: getCategories returned no 'Treatment' column -- no labels available.")
        return {}

    labels: dict[Well, str] = {}
    for plate, well, treatment in zip(df_labels["Plate"], df_labels["Well"], df_labels["Treatment"]):
        if treatment is None or pd.isna(treatment) or str(treatment) == "":
            continue
        labels[(plate, well)] = str(treatment)
    return labels


def resolve_labels_from_csv(
    wells: list[Well],
    csv_path: str,
    plate_col: str = "Plate",
    well_col: str = "Well",
    label_col: str = "label",
) -> dict[Well, str]:
    """Look up labels for (plate, well) pairs in a ``Plate,Well,label`` CSV.

    The CSV is a flat lookup table, not a database: it's read in full each
    call rather than cached, which is fine at the scale this project operates
    at (a handful of plates) and keeps this function stateless like its
    MongoDB/dummy counterparts.

    Only wells present in both ``wells`` and the CSV, with a non-null/non-empty
    label, are returned -- matching ``resolve_labels_from_mongodb``'s
    "missing/blank labels are dropped, not propagated" contract.
    """
    import pandas as pd

    if not wells:
        return {}

    df = pd.read_csv(csv_path, dtype={plate_col: str, well_col: str})
    missing_cols = {plate_col, well_col, label_col} - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"CSV {csv_path!r} is missing expected column(s): {sorted(missing_cols)} "
            f"(found: {list(df.columns)})"
        )

    wanted = set(wells)
    labels: dict[Well, str] = {}
    for plate, well, label in zip(df[plate_col], df[well_col], df[label_col]):
        key = (plate, well)
        if key not in wanted:
            continue
        if label is None or pd.isna(label) or str(label) == "":
            continue
        labels[key] = str(label)
    return labels


def resolve_labels_dummy(
    wells: list[Well],
    class_names: list[str] | tuple[str, ...] | None = None,
    seed: int = 42,
) -> dict[Well, str]:
    """Assign deterministic pseudo-random labels, for tests/offline runs.

    Wells are sorted before assignment so the mapping depends only on the well
    set and the seed, never on filesystem iteration order.
    """
    import numpy as np

    class_names = tuple(class_names) if class_names else DEFAULT_DUMMY_CLASSES
    rng = np.random.RandomState(seed)
    return {well: class_names[rng.randint(0, len(class_names))] for well in sorted(wells)}


class MongoDBLabelsProvider:
    """Label provider using MongoDB.

    Queries MongoDB to get labels for plate/well combinations.
    """

    def __init__(self, collection: str = "tags"):
        self.collection = collection

    def get_labels_for_wells(
        self,
        wells: list[Well] | set[Well],
    ) -> dict[Well, str]:
        """Query MongoDB for the given wells' treatments."""
        return resolve_labels_from_mongodb(sorted(wells), collection=self.collection)

    def get_labels(
        self,
        root_dir: str,
        exclude_wells: list[Well] | None = None,
    ) -> dict[Well, str]:
        """Discover wells under ``root_dir`` and query MongoDB for their treatments.

        Args:
            root_dir: Root directory containing images.
            exclude_wells: List of (plate, well) tuples to exclude (corrupted images).
        """
        from src.sample_selection import scan_directory, wells_from_index

        wells = wells_from_index(scan_directory(root_dir), exclude_wells=exclude_wells)
        return self.get_labels_for_wells(wells)


class CSVLabelsProvider:
    """Label provider reading a ``Plate,Well,<label column>`` CSV lookup file."""

    def __init__(
        self,
        csv_path: str,
        plate_col: str = "Plate",
        well_col: str = "Well",
        label_col: str = "label",
    ):
        self.csv_path = csv_path
        self.plate_col = plate_col
        self.well_col = well_col
        self.label_col = label_col

    def get_labels_for_wells(
        self,
        wells: list[Well] | set[Well],
    ) -> dict[Well, str]:
        """Look up the given wells' labels in the CSV."""
        return resolve_labels_from_csv(
            sorted(wells), self.csv_path,
            plate_col=self.plate_col, well_col=self.well_col, label_col=self.label_col,
        )

    def get_labels(
        self,
        root_dir: str,
        exclude_wells: list[Well] | None = None,
    ) -> dict[Well, str]:
        """Discover wells under ``root_dir`` and look up their labels in the CSV.

        Args:
            root_dir: Root directory containing images.
            exclude_wells: List of (plate, well) tuples to exclude (corrupted images).
        """
        from src.sample_selection import scan_directory, wells_from_index

        wells = wells_from_index(scan_directory(root_dir), exclude_wells=exclude_wells)
        return self.get_labels_for_wells(wells)


class DummyLabelsProvider:
    """Dummy label provider for testing without MongoDB/CSV.

    Generates deterministic pseudo-random string labels based on plate/well
    combinations. Not a real label source -- for tests/offline dry runs only.
    """

    DEFAULT_CLASSES: tuple[str, ...] = DEFAULT_DUMMY_CLASSES

    def __init__(self, class_names: list[str] | None = None, seed: int = 42):
        self.class_names = class_names or self.DEFAULT_CLASSES
        self.seed = seed

    def get_labels_for_wells(
        self,
        wells: list[Well] | set[Well],
    ) -> dict[Well, str]:
        """Assign deterministic pseudo-random labels to the given wells."""
        return resolve_labels_dummy(sorted(wells), class_names=self.class_names, seed=self.seed)

    def get_labels(
        self,
        root_dir: str,
        exclude_wells: list[Well] | None = None,
    ) -> dict[Well, str]:
        """Discover wells under ``root_dir`` and label them.

        Args:
            root_dir: Root directory containing images.
            exclude_wells: List of (plate, well) tuples to exclude (corrupted images).
        """
        from src.sample_selection import scan_directory, wells_from_index

        wells = wells_from_index(scan_directory(root_dir), exclude_wells=exclude_wells)
        return self.get_labels_for_wells(wells)


def limit_wells_per_label(
    labels: dict[Well, str],
    n_per_label: int | None,
    verbose: bool = False,
) -> dict[Well, str]:
    """Keep only the first N wells per label class, for a consistent subset.

    Wells are sorted within each label before truncating, so the selected
    subset is stable across runs and machines. ``None`` means no limit.
    """
    if n_per_label is None:
        return labels

    from collections import defaultdict

    wells_by_label: dict[str, list[Well]] = defaultdict(list)
    for well, label in labels.items():
        wells_by_label[label].append(well)

    limited: dict[Well, str] = {}
    for label, wells in wells_by_label.items():
        for well in sorted(wells)[:n_per_label]:
            limited[well] = label

    print(f"Limited to {n_per_label} wells per label: {len(labels)} -> {len(limited)} wells")
    if verbose:
        for label in sorted(wells_by_label):
            selected = sorted(wells_by_label[label])[:n_per_label]
            print(f"  {label}: {[f'{plate}/{well}' for plate, well in selected]}")

    return limited
