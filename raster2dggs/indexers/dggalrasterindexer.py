"""
@author: alpha-beta-soup
"""

from abc import abstractmethod
from functools import reduce, lru_cache
from numbers import Number
from typing import List, Tuple

import dggal
import pandas as pd
import pyarrow as pa
import xarray as xr
import numpy as np
import shapely

import raster2dggs.constants as const

from raster2dggs.indexers.rasterindexer import RasterIndexer

# Instantiate DGGAL
dggal.pydggal_setup(dggal.Application(appGlobals=globals()))


class DGGALRasterIndexer(RasterIndexer):
    """
    Provides integration for DGGRSs depending on the DGGAL API.
    """

    @property
    @abstractmethod
    def dggrs(self) -> dggal.DGGRS:
        raise NotImplementedError

    @lru_cache(maxsize=None)
    def _get_parent(self, zone: int) -> int:
        """
        Get immediate parent with caching.
        Used recursively, the LRU cache will naturally evict leaf cells which don't benefit from caching.
        """
        # NB  All zones of GNOSIS Global Grid and ISEA9R have single parents, whereas ISEA3H zones have one parent if they are a centroid child, and three parents otherwise if they are a vertex child.  See dggrs.getMaxParents()
        parents = self.dggrs.getZoneParents(zone)
        if self.dggrs.getMaxParents() == 1:
            return parents[0]
        # Find centroid parent for multi-parent DGGRS
        return next(
            (p for p in parents if self.dggrs.isZoneCentroidChild(p)), parents[0]
        )

    def _get_ancestor(self, zone: int, levels_up: int = 1) -> int:
        """Get ancestor by repeatedly calling cached parent lookup."""
        parent = zone
        for _ in range(levels_up):
            parent = self._get_parent(int(parent))
        return parent

    def index_func(
        self,
        sdf: xr.DataArray,
        resolution: int,
        parent_res: int,
        nodata: Number = np.nan,
        band_labels: Tuple[str] = None,
    ) -> pa.Table:
        """
        Index a raster window to a DGGRS.
        Subsequent steps are necessary to resolve issues at the boundaries of windows.
        If windows are very small, or in strips rather than blocks, processing may be slower
        than necessary and the recommendation is to write different windows in the source raster.

        Implementation of interface function.
        """
        sdf: pd.DataFrame = (
            sdf.to_dataframe().drop(columns=["spatial_ref"]).reset_index()
        )
        subset: pd.DataFrame = sdf.dropna()
        subset = subset[subset.value != nodata]
        subset = pd.pivot_table(
            subset, values=const.DEFAULT_NAME, index=["x", "y"], columns=["band"]
        ).reset_index()
        # Primary DGGSRS index
        cells = [
            self.dggrs.getZoneFromWGS84Centroid(resolution, dggal.GeoPoint(lon, lat))
            for lon, lat in zip(subset["y"], subset["x"])
        ]  # Vectorised
        dggrs_parent = [
            self._get_ancestor(zone, resolution - parent_res) for zone in cells
        ]
        subset = subset.drop(columns=["x", "y"])
        index_col = self.index_col(resolution)
        subset[index_col] = pd.Series(
            map(self.dggrs.getZoneTextID, cells), index=subset.index
        )
        partition_col = self.partition_col(parent_res)
        subset[partition_col] = pd.Series(
            map(self.dggrs.getZoneTextID, dggrs_parent), index=subset.index
        )
        # Rename bands
        bands = sdf["band"].unique()
        columns = dict(zip(bands, band_labels))
        subset = subset.rename(columns=columns)
        return pa.Table.from_pandas(subset)

    def cell_to_children_size(self, cell: str, desired_resolution: int) -> int:
        """
        Determine total number of children at some offset resolution

        Implementation of interface function.
        """
        current_resolution = self.dggrs.getZoneLevel(self.dggrs.getZoneFromTextID(cell))
        n = desired_resolution - current_resolution
        return self.dggrs.getRefinementRatio() ** n

    @staticmethod
    def valid_set(cells: set) -> set[str]:
        """
        Implementation of interface function.
        """
        return set(filter(lambda c: (not pd.isna(c)), cells))

    def parent_cells(self, cells: set, resolution) -> map:
        """
        Implementation of interface function.
        """
        # TODO appropriately handle potential for multiple parentage in dggal (e.g. ISEAH3)
        child_resolution = self.dggrs.getZoneLevel(
            self.dggrs.getZoneFromTextID(next(iter(cells)))
        )
        return map(
            lambda zone: self.dggrs.getZoneTextID(
                self._get_ancestor(
                    self.dggrs.getZoneFromTextID(zone), child_resolution - resolution
                )
            ),
            cells,
        )

    def expected_count(self, parent: str, resolution: int):
        """
        Implementation of interface function.
        """
        return self.cell_to_children_size(parent, resolution)

    def cell_to_point(self, cell: str) -> shapely.geometry.Point:
        geo_point: dggal.GeoPoint = self.dggrs.getZoneWGS84Centroid(
            self.dggrs.getZoneFromTextID(cell)
        )
        return shapely.Point(geo_point.lon, geo_point.lat)

    def cell_to_polygon(
        self, cell: str, edgeRefinement: int = 0
    ) -> shapely.geometry.Polygon:
        geo_points: List[dggal.GeoPoint] = self.dggrs.getZoneRefinedWGS84Vertices(
            self.dggrs.getZoneFromTextID(cell), edgeRefinement
        )
        return shapely.Polygon(tuple([(p.lon, p.lat) for p in geo_points]))


class ISEA4RRasterIndexer(DGGALRasterIndexer):
    """
    A raster indexer for the ISEA4R DGGS, an equal area rhombic grid with a refinement ratio of 4 defined in the ISEA projection transformed into a 5x6 Cartesian space resulting in axis-aligned square zones.
    """

    def __init__(self, dggs: str):
        super().__init__(dggs)
        self._dggrs = dggal.ISEA4R()

    @property
    def dggrs(self) -> dggal.DGGRS:
        return self._dggrs


class IVEA4RRasterIndexer(DGGALRasterIndexer):
    """
    A raster indexer for the IVEA4R DGGS, an equal area rhombic grid with a refinement ratio of 4 defined in the IVEA projection transformed into a 5x6 Cartesian space resulting in axis-aligned square zones, using the same global indexing and sub-zone ordering as for ISEA4R.
    """

    def __init__(self, dggs: str):
        super().__init__(dggs)
        self._dggrs = dggal.IVEA4R()

    @property
    def dggrs(self) -> dggal.DGGRS:
        return self._dggrs


class ISEA9RRasterIndexer(DGGALRasterIndexer):
    """
    A raster indexer for the ISEA9R DGGS, an equal area rhombic grid with a refinement ratio of 9 defined in the ISEA projection transformed into a 5x6 Cartesian space resulting in axis-aligned square zones.
    """

    def __init__(self, dggs: str):
        super().__init__(dggs)
        self._dggrs = dggal.ISEA9R()

    @property
    def dggrs(self) -> dggal.DGGRS:
        return self._dggrs


class IVEA9RRasterIndexer(DGGALRasterIndexer):
    """
    A raster indexer for the ISEA9R DGGS, an equal area rhombic grid with a refinement ratio of 9 defined in the IVEA projection transformed into a 5x6 Cartesian space resulting in axis-aligned square zones.
    """

    def __init__(self, dggs: str):
        super().__init__(dggs)
        self._dggrs = dggal.IVEA9R()

    @property
    def dggrs(self) -> dggal.DGGRS:
        return self._dggrs


class ISEA3HRasterIndexer(DGGALRasterIndexer):
    """
    A raster indexer for the ISEA3H DGGS, an equal area hexagonal grid with a refinement ratio of 3 defined in the ISEA projection.
    """

    def __init__(self, dggs: str):
        super().__init__(dggs)
        self._dggrs = dggal.ISEA3H()

    @property
    def dggrs(self) -> dggal.DGGRS:
        return self._dggrs


class IVEA3HRasterIndexer(DGGALRasterIndexer):
    """
    A raster indexer for the IVEA3H DGGS, an equal area hexagonal grid with a refinement ratio of 3 defined in the IVEA projection, using the same global indexing and sub-zone ordering as for ISEA3H.
    """

    def __init__(self, dggs: str):
        super().__init__(dggs)
        self._dggrs = dggal.IVEA3H()

    @property
    def dggrs(self) -> dggal.DGGRS:
        return self._dggrs


class ISEA7HRasterIndexer(DGGALRasterIndexer):
    """
    A raster indexer for the ISEA7H DGGS, an equal area hexagonal grid with a refinement ratio of 7 defined in the ISEA projection.
    """

    def __init__(self, dggs: str):
        super().__init__(dggs)
        self._dggrs = dggal.ISEA7H()

    @property
    def dggrs(self) -> dggal.DGGRS:
        return self._dggrs


class IVEA7HRasterIndexer(DGGALRasterIndexer):
    """
    A raster indexer for the IVEA7H DGGS, an equal area hexagonal grid with a refinement ratio of 7 defined in the ISEA projection.
    """

    def __init__(self, dggs: str):
        super().__init__(dggs)
        self._dggrs = dggal.IVEA7H()

    @property
    def dggrs(self) -> dggal.DGGRS:
        return self._dggrs


class ISEA7HZ7RasterIndexer(DGGALRasterIndexer):
    """
    A raster indexer for the ISEA7H DGGS, which has the same Discrete Global Grid Hierarchy (DGGH) and sub-zone order as ISEA7H, but using the Z7 indexing for interoperability with DGGRID and IGEO7.
    """

    def __init__(self, dggs: str):
        super().__init__(dggs)
        self._dggrs = dggal.ISEA7H_Z7()

    @property
    def dggrs(self) -> dggal.DGGRS:
        return self._dggrs


class IVEA7HZ7RasterIndexer(DGGALRasterIndexer):
    """
    A raster indexer for the ISEA7H DGGS, which has the same Discrete Global Grid Hierarchy (DGGH) and sub-zone order as ISEA7H, but using the Z7 indexing for interoperability with DGGRID and IGEO7.
    """

    def __init__(self, dggs: str):
        super().__init__(dggs)
        self._dggrs = dggal.IVEA7H_Z7()

    @property
    def dggrs(self) -> dggal.DGGRS:
        return self._dggrs


class RTEA4RRasterIndexer(DGGALRasterIndexer):
    """
    A raster indexer for the RTEA4R DGGS, an equal-area rhombic grid with a refinement ratio of 4 defined in the RTEA projection transformed into a 5x6 Cartesian space resulting in axis-aligned square zones, using the same global indexing and sub-zone ordering as for ISEA4R.
    """

    def __init__(self, dggs: str):
        super().__init__(dggs)
        self._dggrs = dggal.RTEA4R()

    @property
    def dggrs(self) -> dggal.DGGRS:
        return self._dggrs


class RTEA9RRasterIndexer(DGGALRasterIndexer):
    """
    A raster indexer for the RTEA9R DGGS, an equal-area rhombic grid with a refinement ratio of 9 defined in the RTEA projection transformed into a 5x6 Cartesian space resulting in axis-aligned square zones, using the same global indexing and sub-zone ordering as for ISEA9R.
    """

    def __init__(self, dggs: str):
        super().__init__(dggs)
        self._dggrs = dggal.RTEA9R()

    @property
    def dggrs(self) -> dggal.DGGRS:
        return self._dggrs


class RTEA3HRasterIndexer(DGGALRasterIndexer):
    """
    A raster indexer for the RTEA3H DGGS, an equal area hexagonal grid with a refinement ratio of 3 defined in the RTEA projection using the same global indexing and sub-zone ordering as for ISEA3H.
    """

    def __init__(self, dggs: str):
        super().__init__(dggs)
        self._dggrs = dggal.RTEA3H()

    @property
    def dggrs(self) -> dggal.DGGRS:
        return self._dggrs


class RTEA7HRasterIndexer(DGGALRasterIndexer):
    """
    A raster indexer for the RTEA7H DGGS, an equal-area hexagonal grid with a refinement ratio of 7 defined in the RTEA projection transformed using the same global indexing and sub-zone ordering as for ISEA7H.
    """

    def __init__(self, dggs: str):
        super().__init__(dggs)
        self._dggrs = dggal.RTEA7H()

    @property
    def dggrs(self) -> dggal.DGGRS:
        return self._dggrs


class RTEA7HZ7RasterIndexer(DGGALRasterIndexer):
    """
    A raster indexer for the RTEA7H DGGS, an equal-area hexagonal grid with a refinement ratio of 7 defined in the RTEA projection transformed using the same global indexing and sub-zone ordering as for ISEA7H.
    """

    def __init__(self, dggs: str):
        super().__init__(dggs)
        self._dggrs = dggal.RTEA7H_Z7()

    @property
    def dggrs(self) -> dggal.DGGRS:
        return self._dggrs


class HEALPixRasterIndexer(DGGALRasterIndexer):
    """
    A raster indexer for the HEALPix DGGS, an equal area and axis-aligned grid with square zones topology and a refinement ratio of 4 defined in the HEALPix projection, using configuration Nφ/H = 4, Nθ/K = 3 (same as default PROJ implementation), the new indexing described in OGC API - DGGS Annex B, and scanline-based sub-zone ordering.
    """

    def __init__(self, dggs: str):
        super().__init__(dggs)
        self._dggrs = dggal.HEALPix()

    @property
    def dggrs(self) -> dggal.DGGRS:
        return self._dggrs


class RHEALPixRasterIndexer(DGGALRasterIndexer):
    """
    A raster indexer for the HEALPix DGGS, an equal area and axis-aligned grid with square zones topology and a refinement ratio of 9 defined in the rHEALPix projection using 50° E prime meridian (equivalent to PROJ implementation with parameters +proj=rhealpix +lon_0=50 +ellps=WGS84), the original hierarchical indexing, and scanline-based sub-zone ordering.
    """

    def __init__(self, dggs: str):
        super().__init__(dggs)
        self._dggrs = dggal.rHEALPix()

    @property
    def dggrs(self) -> dggal.DGGRS:
        return self._dggrs
