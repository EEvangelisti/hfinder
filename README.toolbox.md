# HFinder Toolbox

HFinder is distributed with auxiliary scripts that enable visualisation of results and quantitative analyses. These scripts are available in the `toolbox` folder.

## Visualizing Predictions - `annot2images.py`

<p align=center>
<img src="doc/FigVisualisation.png" width="650"/>
</p>

Once predictions have been consolidated into COCO-style JSON files, they can be 
easily visualized with the `annot2images.py` utility. This script loads the JSON 
annotations and the corresponding TIFF channels, then generates PNG images that 
overlay the predicted objects.

Key features:
- **Channel-aware visualization**: Each annotation is shown on the specific 
  channel where it was detected.
- **Bounding boxes & polygons**: Both are drawn in overlay, with optional 
  semi-transparency for polygons to reveal underlying structures.
- **Confidence display**: Prediction confidences are displayed alongside class 
  names; optionally, bounding boxes can be color-coded using a confidence-based 
  palette (e.g., *viridis*).
- **Category-wise output**: One PNG per predicted category is generated, 
  facilitating inspection of individual structures.

This visualization step is particularly useful for quality control, 
publication-ready figures, and quickly identifying whether predicted channels 
match expected biological structures.

|Command|Description|Default value|
|-|-|-|
|`-t <path>` or<br>`--tiff_dir <path>`|Folder containing TIFF files|Current directory|
|`-a <path>` or<br>`--annotations <path>`|Folder containing COCO JSON annotations|Current directory|
|`-o <path>` or<br>`--out_dir <path>`|Output directory for PNG files|Current directory|
|`-lab` or<br>`--no_labels`|Do not display labels and confidence values|Inactive|
|`-box` or<br>`--no_bounding_boxes`|Do not display bounding boxes around polygons|Inactive|
|`-pal <name>` or<br>`--palette <name>`|Matplotlib colormap used to encode confidence values|#00FFFF|
|`-ttf <name>` or<br>`--font_file <name>`|Font used to write labels and confidence values|arial.ttf|
|`-sz <int>` or<br>`--font_size <int>`|Font size for labels and confidence values|Proportional|
|`-long` or<br>`--long_labels`|Do not abbreviate label names|Inactive|
|`-cat <name>` or<br>`--category <name>`|Process the given category only|All categories|



## Signal Extraction - `annot2signal.py`

<p align=center>
<img src="doc/SignalQuantification.png" width="650"/>
</p>

Beyond visualization, it is often essential to measure whether predicted 
structures show a higher-than-expected signal on their corresponding microscopy 
channel. The `annot2signal.py` utility automates this quantification. It 
loads the COCO-style JSON annotations together with the associated TIFF channels, 
extracts per-polygon intensities, and normalizes them against the channel 
background. The result is a distribution of enrichment values, summarized as a 
box plot.  

**Key features:**  

- **Channel-aware quantification:** Each polygon is measured strictly on the channel where it was detected.  
- **Background normalization:** Mean intensities are normalized against the average background of the same channel, with annotated regions excluded to avoid bias.  
- **Thresholding support:** By default, Otsu thresholding is applied to separate true signal from background; alternatively, a fixed threshold can be provided.  
- **Per-polygon statistics:** Every annotation is treated independently, enabling detailed downstream analyses (values can optionally be exported as a TSV).  
- **Publication-ready plots:** Results are visualized as box plots with individual points overlaid, highlighting both distribution and variability.  

This step is particularly useful to assess whether detected objects correspond 
to biologically enriched structures, compare categories or channels 
quantitatively, and provide statistical summaries alongside qualitative overlays.

|Command|Description|Default value|
|-|-|-|
|`-t <path>` or<br>`--tiff_dir <path>`|Folder containing TIFF files|Current directory|
|`-a <path>` or<br>`--annotations <path>`|Folder containing COCO JSON files|Current directory|
|`-o <path>` or<br>`--out_dir <path>`|Output directory for PNG files|Current directory|
|`-cat <name>` or<br>`--category <name>`|Category to analyse|None|
|`-sig` or<br>`--signal`|Index of the channel used to retrieve signal. 'same' = use the detection channel|same|



## Measuring Distances - `annot2distances.py`

Quantifying spatial relationships between annotated objects is useful in microscopy.
The `annot2distances.py` utility automates this process. It takes COCO-style 
JSON annotations and their associated TIFF images, computes object centroids, 
and measures Euclidean distances either between categories (A→B) or within a 
single category (A↔A). Results are exported as a CSV table and complemented by 
optional image overlays for visual inspection.

Key features:

- Centroid-based distances: Distances are calculated between true polygon centroids, not bounding boxes.  

- Flexible pairing: Supports both inter-class (A→B) and intra-class (A↔A) measurements.  

- Multi-dimensional image support: Handles 2D images, multi-channel arrays (C,H,W), and Z-stacks (Z,C,H,W).  

- Automatic file matching: TIFFs and JSONs are linked by basename; multiple JSONs per image are supported.  

- Readable outputs: Results are written to distances.csv, ready for downstream analysis in R, Python, or Excel.  

- Visual validation: PNG overlays display composite images, annotation contours, centroids, and (if enabled) connecting lines between paired objects.  

- Configurable display: Channel coloring, label size, and overlay limits can be adjusted to match dataset needs.  

- Robust error handling: Clear messages are raised for missing annotations, out-of-range channels, or Z-slice mismatches.  

This step is particularly useful for studying spatial organization in microscopy datasets, testing whether two object classes preferentially co-localize, or generating visual evidence to support quantitative summaries.

|Command|Description|Default value|
|-|-|-|
|`-t <path>` or<br>`--tiff_dir <path>`|Folder containing TIFF files|Current directory|
|`-a <path>` or<br>`--annotations <path>`|Folder containing COCO JSON files|Current directory|
|`-o <path>` or<br>`--out_dir <path>`|Output directory for PNG files|Current directory|
|`-clA <name/index>` or<br>`--class_A <name/index>`|Primary class name or numeric id|**No default**|
|`-clB <name/index>` or<br>`--class_B <name/index>`|Secondary class name or numeric id (if omitted, within-class A↔A)|None|
|`-hue <csv>` or<br>`--hues_list <csv>`|Comma-separated hues (degrees) assigned to channels|180,300,0|
|`-s <int>` or<br>`--stroke <int>`|Stroke width for drawing overlays|2|
|`-ttf <file>` or<br>`--font_file <file>`|TTF font for labels (fallback to default if missing)|"DejaVuSans.ttf"|
|`-sz <int>` or<br>`--font_size <int>`|Font size for labels|14|
|`-max <int>` or<br>`--max_png <int>`|Max number of PNG overlays to save (to avoid huge dumps)|24|
|`-lab` or<br>`--no_labels`|Do not show labels|Inactive|

