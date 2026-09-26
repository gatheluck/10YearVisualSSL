# 1. ImageNet-1k

You are performing a controlled ImageNet-1k image classification evaluation.

The uploaded ZIP contains:

- `manifest.json`
- `imagenet1k_vocabulary.json`
- 100 images

Your task is to classify **every image listed in `manifest.json`** into exactly one class from the supplied ImageNet-1k vocabulary.

## Rules

- Use only the supplied images and `imagenet1k_vocabulary.json`.
- Evaluate every sample exactly once.
- Preserve each `sample_id` exactly as written in `manifest.json`.
- Process and return samples in the same order as `manifest.json`.
- Select exactly one allowed ImageNet-1k class for each image.
- Use the exact `class_id` defined in the supplied vocabulary.
- Do not use web search, browsing, external retrieval, external tools, or additional data sources.
- Do not explain your reasoning.
- Do not provide alternative predictions.
- Do not skip uncertain samples.
- If uncertain, return your best single prediction.
- Return only valid JSON. Do not use Markdown code fences.

## Output format

{
  "dataset": "ImageNet-1k",
  "predictions": [
    {
      "sample_id": "<exact sample_id>",
      "class_id": <integer>
    }
  ]
}

## Final internal checks

Before returning the output, internally verify that:

1. the number of predictions equals the number of samples in `manifest.json`;
2. every input `sample_id` appears exactly once;
3. no unknown `sample_id` appears;
4. every `class_id` exists in `imagenet1k_vocabulary.json`;
5. the prediction order matches `manifest.json`;
6. the output is syntactically valid JSON.

Return only the final JSON.


# 2. COCO

You are performing a controlled visual object-category recognition evaluation on COCO.

The uploaded ZIP contains:

- `manifest.json`
- `coco80_vocabulary.json`
- 100 images

Your task is to identify **all visible COCO object categories** present in every image listed in `manifest.json`.

This is a category-presence recognition task, not a bounding-box detection task.

## Rules

- Use only the supplied images and `coco80_vocabulary.json`.
- Evaluate every sample exactly once.
- Preserve each `sample_id` exactly as written in `manifest.json`.
- Process and return samples in the same order as `manifest.json`.
- Use only category IDs that exist in the supplied COCO-80 vocabulary.
- Report a category only when there is visible evidence that at least one instance of that category is present.
- Report each category at most once per image.
- Do not return bounding boxes.
- Do not return segmentation masks.
- Do not use web search, browsing, external retrieval, external tools, or additional data sources.
- Do not explain your reasoning.
- If no COCO category is visible, return an empty list.
- Return only valid JSON. Do not use Markdown code fences.

## Output format

{
  "dataset": "COCO",
  "predictions": [
    {
      "sample_id": "<exact sample_id>",
      "category_ids": [1, 3, 18]
    }
  ]
}

The `category_ids` list may contain zero or more category IDs.

## Final internal checks

Before returning the output, internally verify that:

1. there is exactly one prediction record per input sample;
2. every input `sample_id` appears exactly once;
3. no unknown `sample_id` appears;
4. every returned category ID exists in `coco80_vocabulary.json`;
5. no category ID is duplicated within one image;
6. the prediction order matches `manifest.json`;
7. the output is syntactically valid JSON.

Return only the final JSON.


# 3. ADE20K

You are performing a controlled sparse semantic segmentation evaluation on ADE20K.

The uploaded ZIP contains:

- `manifest.json`
- `ade20k_vocabulary.json`
- 100 RGB images

Each sample in `manifest.json` contains **32 normalized query points**.

Your task is to predict the semantic class located at each specified point.

## Coordinate convention

For every query point:

- `x` is the horizontal coordinate normalized to [0,1];
- `y` is the vertical coordinate normalized to [0,1];
- `(0,0)` is the top-left of the image;
- `(1,1)` is the bottom-right.

Predict the semantic class of the visible region located **at the specified coordinate**, not the dominant class of the whole image.

## Rules

- Use only the supplied RGB image, query coordinates, and `ade20k_vocabulary.json`.
- Evaluate every sample exactly once.
- Evaluate every query point exactly once.
- Preserve every `sample_id` exactly.
- Preserve every `point_id` exactly.
- Process samples in the same order as `manifest.json`.
- Process points in the same order as listed for each sample.
- Return exactly one allowed ADE20K semantic `class_id` per query point.
- Use only class IDs contained in the supplied ADE20K vocabulary.
- Do not use web search, browsing, external retrieval, external tools, or additional data sources.
- Do not explain your reasoning.
- Do not omit uncertain points.
- If uncertain, return your best single semantic class prediction.
- Return only valid JSON. Do not use Markdown code fences.

## Output format

{
  "dataset": "ADE20K",
  "predictions": [
    {
      "sample_id": "<exact sample_id>",
      "point_class_ids": [
        <class_id for point_id 0>,
        <class_id for point_id 1>,
        <class_id for point_id 2>
      ]
    }
  ]
}

The position of each integer in `point_class_ids` must correspond exactly to the ordered `points` array in `manifest.json`.

For example, if a sample contains 32 points, `point_class_ids` must contain exactly 32 integers.

## Final internal checks

Before returning the output, internally verify that:

1. there is exactly one prediction record per input image;
2. every `sample_id` appears exactly once;
3. every prediction contains exactly the same number of class IDs as query points in the corresponding manifest entry;
4. every returned `class_id` exists in `ade20k_vocabulary.json`;
5. sample order matches `manifest.json`;
6. point order matches the manifest exactly;
7. the output is syntactically valid JSON.

Return only the final JSON.



# 4. NYUv2

You are performing a controlled monocular relative-depth evaluation on NYUv2.

The uploaded ZIP contains:

- `manifest.json`
- 100 RGB images

Each sample contains **20 pairs of normalized image coordinates**.

For each pair, your task is to determine which point is physically closer to the camera based only on the RGB image.

## Coordinate convention

For every point:

- `x` is horizontal position normalized to [0,1];
- `y` is vertical position normalized to [0,1];
- `(0,0)` is the top-left;
- `(1,1)` is the bottom-right.

Each pair contains point `a` and point `b`.

## Allowed answers

Use exactly one of the following integer codes:

- `0` = A is closer to the camera
- `1` = B is closer to the camera
- `2` = approximately equal depth
- `3` = genuinely impossible to determine from the supplied image

Use `3` only when the visual evidence is genuinely insufficient. Prefer your best visual depth judgment when possible.

## Rules

- Use only the supplied RGB image and the coordinate pairs.
- Evaluate every sample exactly once.
- Evaluate every pair exactly once.
- Preserve each `sample_id` exactly.
- Process samples in the same order as `manifest.json`.
- Process pairs in the same order as listed in the manifest.
- Do not use web search, browsing, external retrieval, external tools, or additional data sources.
- Do not explain your reasoning.
- Do not output estimated metric depth.
- Return only valid JSON. Do not use Markdown code fences.

## Output format

{
  "dataset": "NYUv2",
  "predictions": [
    {
      "sample_id": "<exact sample_id>",
      "pair_answers": [
        0,
        1,
        0,
        2
      ]
    }
  ]
}

The position of each integer in `pair_answers` corresponds exactly to the ordered pair list in `manifest.json`.

For a sample containing 20 pairs, `pair_answers` must contain exactly 20 integers.

## Final internal checks

Before returning the output, internally verify that:

1. there is exactly one prediction record per input image;
2. every input `sample_id` appears exactly once;
3. every sample contains exactly one answer per supplied pair;
4. every answer is one of `0`, `1`, `2`, or `3`;
5. sample order matches `manifest.json`;
6. pair order matches the manifest exactly;
7. the output is syntactically valid JSON.

Return only the final JSON.


# 5. Something-Something V2 (SSv2)

You are performing a controlled temporal action-recognition evaluation on Something-Something V2.

The uploaded ZIP contains:

- `manifest.json`
- `ssv2_vocabulary.json`
- 100 video samples represented by ordered frames

Each video sample contains **16 frames sampled in chronological order**.

Your task is to classify every sample into exactly one allowed Something-Something V2 action class.

## Important temporal interpretation

The 16 frames belonging to one sample represent a single video.

They are ordered from earliest to latest.

You must interpret them as an ordered temporal sequence, not as 16 independent images.

Pay particular attention to:

- object motion;
- hand-object interaction;
- changes of state;
- direction of motion;
- before/after relationships;
- whether an object approaches, moves away, enters, leaves, covers, uncovers, opens, closes, moves up/down, or changes relative position.

## Rules

- Use only the supplied ordered frames and `ssv2_vocabulary.json`.
- Evaluate every video sample exactly once.
- Preserve each `sample_id` exactly as written in `manifest.json`.
- Process samples in the same order as `manifest.json`.
- Use all 16 frames when forming the prediction.
- Select exactly one action class per sample.
- Use only a `class_id` contained in `ssv2_vocabulary.json`.
- Do not use web search, browsing, external retrieval, external tools, or additional data sources.
- Do not explain your reasoning.
- Do not return alternative action classes.
- Do not omit uncertain samples.
- If uncertain, return your best single action prediction.
- Return only valid JSON. Do not use Markdown code fences.

## Output format

{
  "dataset": "SSv2",
  "predictions": [
    {
      "sample_id": "<exact sample_id>",
      "class_id": <integer>
    }
  ]
}

## Final internal checks

Before returning the output, internally verify that:

1. there is exactly one prediction for every sample in `manifest.json`;
2. every input `sample_id` appears exactly once;
3. no unknown `sample_id` appears;
4. every `class_id` exists in `ssv2_vocabulary.json`;
5. all 16 frames for each sample were treated as one chronological sequence;
6. the prediction order matches `manifest.json`;
7. the output is syntactically valid JSON.

Return only the final JSON.
