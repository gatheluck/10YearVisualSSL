// Figure 3, page 4. Strings retain the manuscript precision.
const results = [
  {
    "model": "DINOv3-7B",
    "task": "ImageNet-1K",
    "protocol": "LP",
    "mean": "86.54",
    "sd": "0.01",
    "short": false
  },
  {
    "model": "DINOv3-7B",
    "task": "ImageNet-1K",
    "protocol": "AP",
    "mean": "87.82",
    "sd": "0.01",
    "short": false
  },
  {
    "model": "DINOv3-7B",
    "task": "ImageNet-1K",
    "protocol": "LoRA",
    "mean": "87.90",
    "sd": "0.03",
    "short": true
  },
  {
    "model": "DINOv3-7B",
    "task": "COCO",
    "protocol": "LP",
    "mean": "40.77",
    "sd": "0.04",
    "short": false
  },
  {
    "model": "DINOv3-7B",
    "task": "COCO",
    "protocol": "AP",
    "mean": "43.78",
    "sd": "0.07",
    "short": false
  },
  {
    "model": "DINOv3-7B",
    "task": "COCO",
    "protocol": "LoRA",
    "mean": "52.63",
    "sd": "0.04",
    "short": false
  },
  {
    "model": "DINOv3-7B",
    "task": "ADE20K",
    "protocol": "LP",
    "mean": "47.14",
    "sd": "0.07",
    "short": false
  },
  {
    "model": "DINOv3-7B",
    "task": "ADE20K",
    "protocol": "AP",
    "mean": "47.91",
    "sd": "0.07",
    "short": false
  },
  {
    "model": "DINOv3-7B",
    "task": "ADE20K",
    "protocol": "DPT",
    "mean": "47.39",
    "sd": "0.38",
    "short": false
  },
  {
    "model": "DINOv3-7B",
    "task": "ADE20K",
    "protocol": "LoRA",
    "mean": "51.67",
    "sd": "0.16",
    "short": false
  },
  {
    "model": "DINOv3-7B",
    "task": "NYUv2",
    "protocol": "LP",
    "mean": "0.577",
    "sd": "0.001",
    "short": false
  },
  {
    "model": "DINOv3-7B",
    "task": "NYUv2",
    "protocol": "AP",
    "mean": "0.423",
    "sd": "0.001",
    "short": false
  },
  {
    "model": "DINOv3-7B",
    "task": "NYUv2",
    "protocol": "DPT",
    "mean": "0.401",
    "sd": "0.001",
    "short": false
  },
  {
    "model": "DINOv3-7B",
    "task": "NYUv2",
    "protocol": "LoRA",
    "mean": "0.396",
    "sd": "0.002",
    "short": false
  },
  {
    "model": "DINOv3-7B",
    "task": "SSv2",
    "protocol": "LP",
    "mean": "14.90",
    "sd": "0.23",
    "short": false
  },
  {
    "model": "DINOv3-7B",
    "task": "SSv2",
    "protocol": "AP",
    "mean": "47.43",
    "sd": "0.34",
    "short": false
  },
  {
    "model": "DINOv3-7B",
    "task": "SSv2",
    "protocol": "LoRA",
    "mean": "56.12",
    "sd": "0.11",
    "short": true
  },
  {
    "model": "VGGT-Ω",
    "task": "ImageNet-1K",
    "protocol": "LP",
    "mean": "10.68",
    "sd": "0.02",
    "short": false
  },
  {
    "model": "VGGT-Ω",
    "task": "ImageNet-1K",
    "protocol": "AP",
    "mean": "81.62",
    "sd": "0.07",
    "short": false
  },
  {
    "model": "VGGT-Ω",
    "task": "ImageNet-1K",
    "protocol": "LoRA",
    "mean": "85.49",
    "sd": "0.05",
    "short": false
  },
  {
    "model": "VGGT-Ω",
    "task": "COCO",
    "protocol": "LP",
    "mean": "10.54",
    "sd": "0.09",
    "short": false
  },
  {
    "model": "VGGT-Ω",
    "task": "COCO",
    "protocol": "AP",
    "mean": "11.63",
    "sd": "0.43",
    "short": false
  },
  {
    "model": "VGGT-Ω",
    "task": "COCO",
    "protocol": "LoRA",
    "mean": "42.61",
    "sd": "0.16",
    "short": false
  },
  {
    "model": "VGGT-Ω",
    "task": "ADE20K",
    "protocol": "LP",
    "mean": "30.48",
    "sd": "0.25",
    "short": false
  },
  {
    "model": "VGGT-Ω",
    "task": "ADE20K",
    "protocol": "AP",
    "mean": "27.18",
    "sd": "0.05",
    "short": false
  },
  {
    "model": "VGGT-Ω",
    "task": "ADE20K",
    "protocol": "DPT",
    "mean": "42.86",
    "sd": "0.15",
    "short": false
  },
  {
    "model": "VGGT-Ω",
    "task": "ADE20K",
    "protocol": "LoRA",
    "mean": "44.46",
    "sd": "0.24",
    "short": false
  },
  {
    "model": "VGGT-Ω",
    "task": "NYUv2",
    "protocol": "LP",
    "mean": "0.678",
    "sd": "0.009",
    "short": false
  },
  {
    "model": "VGGT-Ω",
    "task": "NYUv2",
    "protocol": "AP",
    "mean": "0.627",
    "sd": "0.005",
    "short": false
  },
  {
    "model": "VGGT-Ω",
    "task": "NYUv2",
    "protocol": "DPT",
    "mean": "0.425",
    "sd": "0.002",
    "short": false
  },
  {
    "model": "VGGT-Ω",
    "task": "NYUv2",
    "protocol": "LoRA",
    "mean": "0.418",
    "sd": "0.003",
    "short": false
  },
  {
    "model": "VGGT-Ω",
    "task": "SSv2",
    "protocol": "LP",
    "mean": "4.70",
    "sd": "0.02",
    "short": false
  },
  {
    "model": "VGGT-Ω",
    "task": "SSv2",
    "protocol": "AP",
    "mean": "44.57",
    "sd": "0.25",
    "short": false
  },
  {
    "model": "VGGT-Ω",
    "task": "SSv2",
    "protocol": "LoRA",
    "mean": "68.12",
    "sd": "0.10",
    "short": false
  },
  {
    "model": "RAEv2",
    "task": "ImageNet-1K",
    "protocol": "LP",
    "mean": "75.41",
    "sd": "0.01",
    "short": false
  },
  {
    "model": "RAEv2",
    "task": "ImageNet-1K",
    "protocol": "AP",
    "mean": "86.26",
    "sd": "0.11",
    "short": false
  },
  {
    "model": "RAEv2",
    "task": "ImageNet-1K",
    "protocol": "LoRA",
    "mean": "87.34",
    "sd": "0.01",
    "short": false
  },
  {
    "model": "RAEv2",
    "task": "COCO",
    "protocol": "LP",
    "mean": "38.91",
    "sd": "0.12",
    "short": false
  },
  {
    "model": "RAEv2",
    "task": "COCO",
    "protocol": "AP",
    "mean": "41.31",
    "sd": "0.51",
    "short": false
  },
  {
    "model": "RAEv2",
    "task": "COCO",
    "protocol": "LoRA",
    "mean": "47.37",
    "sd": "0.11",
    "short": false
  },
  {
    "model": "RAEv2",
    "task": "ADE20K",
    "protocol": "LP",
    "mean": "43.04",
    "sd": "0.09",
    "short": false
  },
  {
    "model": "RAEv2",
    "task": "ADE20K",
    "protocol": "AP",
    "mean": "45.19",
    "sd": "0.39",
    "short": false
  },
  {
    "model": "RAEv2",
    "task": "ADE20K",
    "protocol": "DPT",
    "mean": "46.24",
    "sd": "0.34",
    "short": false
  },
  {
    "model": "RAEv2",
    "task": "ADE20K",
    "protocol": "LoRA",
    "mean": "46.70",
    "sd": "0.22",
    "short": false
  },
  {
    "model": "RAEv2",
    "task": "NYUv2",
    "protocol": "LP",
    "mean": "0.662",
    "sd": "0.039",
    "short": false
  },
  {
    "model": "RAEv2",
    "task": "NYUv2",
    "protocol": "AP",
    "mean": "0.476",
    "sd": "0.002",
    "short": false
  },
  {
    "model": "RAEv2",
    "task": "NYUv2",
    "protocol": "DPT",
    "mean": "0.436",
    "sd": "0.002",
    "short": false
  },
  {
    "model": "RAEv2",
    "task": "NYUv2",
    "protocol": "LoRA",
    "mean": "0.472",
    "sd": "0.002",
    "short": false
  },
  {
    "model": "RAEv2",
    "task": "SSv2",
    "protocol": "LP",
    "mean": "8.20",
    "sd": "0.37",
    "short": false
  },
  {
    "model": "RAEv2",
    "task": "SSv2",
    "protocol": "AP",
    "mean": "40.69",
    "sd": "0.08",
    "short": false
  },
  {
    "model": "RAEv2",
    "task": "SSv2",
    "protocol": "LoRA",
    "mean": "48.70",
    "sd": "0.15",
    "short": false
  },
  {
    "model": "V-JEPA 2.1-G",
    "task": "ImageNet-1K",
    "protocol": "LP",
    "mean": "57.61",
    "sd": "0.03",
    "short": false
  },
  {
    "model": "V-JEPA 2.1-G",
    "task": "ImageNet-1K",
    "protocol": "AP",
    "mean": "81.39",
    "sd": "0.13",
    "short": false
  },
  {
    "model": "V-JEPA 2.1-G",
    "task": "ImageNet-1K",
    "protocol": "LoRA",
    "mean": "83.88",
    "sd": "0.03",
    "short": false
  },
  {
    "model": "V-JEPA 2.1-G",
    "task": "COCO",
    "protocol": "LP",
    "mean": "35.39",
    "sd": "0.35",
    "short": false
  },
  {
    "model": "V-JEPA 2.1-G",
    "task": "COCO",
    "protocol": "AP",
    "mean": "35.65",
    "sd": "0.28",
    "short": false
  },
  {
    "model": "V-JEPA 2.1-G",
    "task": "COCO",
    "protocol": "LoRA",
    "mean": "46.82",
    "sd": "0.04",
    "short": false
  },
  {
    "model": "V-JEPA 2.1-G",
    "task": "ADE20K",
    "protocol": "LP",
    "mean": "36.10",
    "sd": "0.12",
    "short": false
  },
  {
    "model": "V-JEPA 2.1-G",
    "task": "ADE20K",
    "protocol": "AP",
    "mean": "39.13",
    "sd": "0.12",
    "short": false
  },
  {
    "model": "V-JEPA 2.1-G",
    "task": "ADE20K",
    "protocol": "DPT",
    "mean": "38.26",
    "sd": "0.12",
    "short": false
  },
  {
    "model": "V-JEPA 2.1-G",
    "task": "ADE20K",
    "protocol": "LoRA",
    "mean": "44.99",
    "sd": "0.13",
    "short": false
  },
  {
    "model": "V-JEPA 2.1-G",
    "task": "NYUv2",
    "protocol": "LP",
    "mean": "0.776",
    "sd": "0.029",
    "short": false
  },
  {
    "model": "V-JEPA 2.1-G",
    "task": "NYUv2",
    "protocol": "AP",
    "mean": "0.473",
    "sd": "0.001",
    "short": false
  },
  {
    "model": "V-JEPA 2.1-G",
    "task": "NYUv2",
    "protocol": "DPT",
    "mean": "0.454",
    "sd": "0.001",
    "short": false
  },
  {
    "model": "V-JEPA 2.1-G",
    "task": "NYUv2",
    "protocol": "LoRA",
    "mean": "0.461",
    "sd": "0.003",
    "short": false
  },
  {
    "model": "V-JEPA 2.1-G",
    "task": "SSv2",
    "protocol": "LP",
    "mean": "21.51",
    "sd": "0.22",
    "short": false
  },
  {
    "model": "V-JEPA 2.1-G",
    "task": "SSv2",
    "protocol": "AP",
    "mean": "69.21",
    "sd": "0.17",
    "short": false
  },
  {
    "model": "V-JEPA 2.1-G",
    "task": "SSv2",
    "protocol": "LoRA",
    "mean": "75.72",
    "sd": "0.20",
    "short": false
  },
  {
    "model": "SigLIP 2-G",
    "task": "ImageNet-1K",
    "protocol": "LP",
    "mean": "84.61",
    "sd": "0.01",
    "short": false
  },
  {
    "model": "SigLIP 2-G",
    "task": "ImageNet-1K",
    "protocol": "AP",
    "mean": "87.36",
    "sd": "0.03",
    "short": false
  },
  {
    "model": "SigLIP 2-G",
    "task": "ImageNet-1K",
    "protocol": "LoRA",
    "mean": "87.31",
    "sd": "0.10",
    "short": false
  },
  {
    "model": "SigLIP 2-G",
    "task": "COCO",
    "protocol": "LP",
    "mean": "35.51",
    "sd": "0.06",
    "short": false
  },
  {
    "model": "SigLIP 2-G",
    "task": "COCO",
    "protocol": "AP",
    "mean": "38.67",
    "sd": "0.10",
    "short": false
  },
  {
    "model": "SigLIP 2-G",
    "task": "COCO",
    "protocol": "LoRA",
    "mean": "49.70",
    "sd": "0.09",
    "short": false
  },
  {
    "model": "SigLIP 2-G",
    "task": "ADE20K",
    "protocol": "LP",
    "mean": "37.21",
    "sd": "0.08",
    "short": false
  },
  {
    "model": "SigLIP 2-G",
    "task": "ADE20K",
    "protocol": "AP",
    "mean": "44.09",
    "sd": "0.11",
    "short": false
  },
  {
    "model": "SigLIP 2-G",
    "task": "ADE20K",
    "protocol": "DPT",
    "mean": "43.03",
    "sd": "0.30",
    "short": false
  },
  {
    "model": "SigLIP 2-G",
    "task": "ADE20K",
    "protocol": "LoRA",
    "mean": "49.08",
    "sd": "0.23",
    "short": false
  },
  {
    "model": "SigLIP 2-G",
    "task": "NYUv2",
    "protocol": "LP",
    "mean": "0.848",
    "sd": "0.007",
    "short": false
  },
  {
    "model": "SigLIP 2-G",
    "task": "NYUv2",
    "protocol": "AP",
    "mean": "0.692",
    "sd": "0.004",
    "short": false
  },
  {
    "model": "SigLIP 2-G",
    "task": "NYUv2",
    "protocol": "DPT",
    "mean": "0.480",
    "sd": "0.001",
    "short": false
  },
  {
    "model": "SigLIP 2-G",
    "task": "NYUv2",
    "protocol": "LoRA",
    "mean": "0.479",
    "sd": "0.005",
    "short": false
  },
  {
    "model": "SigLIP 2-G",
    "task": "SSv2",
    "protocol": "LP",
    "mean": "22.96",
    "sd": "1.06",
    "short": false
  },
  {
    "model": "SigLIP 2-G",
    "task": "SSv2",
    "protocol": "AP",
    "mean": "40.90",
    "sd": "0.57",
    "short": false
  },
  {
    "model": "SigLIP 2-G",
    "task": "SSv2",
    "protocol": "LoRA",
    "mean": "48.92",
    "sd": "0.29",
    "short": false
  }
];
