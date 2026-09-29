import { env, pipeline, RawImage } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1";

env.allowLocalModels = false;

const MODEL_ID = "Xenova/clip-vit-base-patch32";
const MAX_IMAGE_SIDE = 640;
const CANDIDATES = [
  {
    label: "室内に見える火や煙",
    prompt: "a photo of visible flames or smoke inside a home",
  },
  {
    label: "ガス漏れ、損傷した容器、熱源に近いガス容器",
    prompt: "a photo showing a gas leak, a damaged pressurized gas cylinder, or a gas cylinder next to a flame or heat source indoors",
  },
  {
    label: "水漏れや配管からの水滴",
    prompt: "a photo showing water leaking or dripping from a pipe inside a home",
  },
  {
    label: "電気設備の近くの水",
    prompt: "a photo showing pooled water touching an electrical outlet or appliance",
  },
  {
    label: "焦げたコンセント、火花、傷んだ配線",
    prompt: "a photo showing a burnt electrical outlet, sparks, or frayed wiring",
  },
  {
    label: "コンセントの過負荷",
    prompt: "a photo showing an overloaded power strip or electrical outlet",
  },
  {
    label: "避難経路をふさぐ物",
    prompt: "a photo showing a blocked emergency exit or obstructed evacuation route",
  },
  {
    label: "明確な兆候を画像から特定できない室内",
    prompt: "a normal home interior with no obvious visible fire, gas, water leak, or electrical hazard",
  },
];

const photoInput = document.querySelector("#photo");
const photoName = document.querySelector("#photo-name");
const preview = document.querySelector("#preview");
const analyzeButton = document.querySelector("#analyze");
const status = document.querySelector("#status");
const suggestions = document.querySelector("#suggestions");
const areas = document.querySelector("#areas");
const level = document.querySelector("#level");
const levelDetail = document.querySelector("#level-detail");
const observationInputs = [...document.querySelectorAll('input[type="checkbox"][data-level]')];

let previewUrl;
let classifier;
let busy = false;

function updateRiskLevel() {
  const checked = observationInputs.filter((input) => input.checked);
  const highestLevel = checked.reduce(
    (highest, input) => Math.max(highest, Number(input.dataset.level)),
    0,
  );

  if (highestLevel === 3) {
    level.className = "risk-level risk-high";
    level.textContent = "危険度の目安：緊急性の高い兆候あり";
    levelDetail.textContent =
      "近づいたり触れたりせず、安全な場所へ離れてください。火・煙・ガス臭・火花がある場合は、地域の緊急窓口へ連絡してください。";
  } else if (highestLevel === 2) {
    level.className = "risk-level risk-medium";
    level.textContent = "危険度の目安：要注意";
    levelDetail.textContent =
      "使用を控え、安全な範囲から確認してください。状況に応じて管理会社や専門業者へ点検を依頼してください。";
  } else if (highestLevel === 1) {
    level.className = "risk-level risk-low";
    level.textContent = "危険度の目安：要確認";
    levelDetail.textContent =
      "過負荷を避け、避難経路を確保してください。ほかの危険の有無も目視で確認してください。";
  } else {
    level.className = "risk-level risk-unrated";
    level.textContent = "危険度の目安：未評価";
    levelDetail.textContent =
      "確認できた兆候を選ぶと目安を表示します。「該当なし」は安全の証明になりません。";
  }
}

function setBusy(value) {
  busy = value;
  photoInput.disabled = value;
  analyzeButton.disabled = value || !photoInput.files?.[0];
}

function progressCallback(event) {
  if (event.status === "progress" && Number.isFinite(event.progress)) {
    const file = event.file ? ` (${event.file})` : "";
    status.textContent = `モデルを読み込み中: ${Math.round(event.progress)}%${file}`;
  }
}

async function requireWebGPU() {
  if (!navigator.gpu) {
    throw new Error("このブラウザではWebGPUが利用できません。対応ブラウザで開いてください。");
  }

  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    throw new Error("WebGPU adapterを取得できませんでした。");
  }
}

async function getClassifier() {
  if (!classifier) {
    classifier = await pipeline(
      "zero-shot-image-classification",
      MODEL_ID,
      {
        device: "webgpu",
        dtype: "q4",
        progress_callback: progressCallback,
      },
    );
  }

  return classifier;
}

async function makeInputCanvas(file) {
  const image = new Image();
  const url = URL.createObjectURL(file);

  try {
    image.src = url;
    await image.decode();

    const scale = Math.min(
      1,
      MAX_IMAGE_SIDE / Math.max(image.naturalWidth, image.naturalHeight),
    );
    const canvas = document.createElement("canvas");
    canvas.width = Math.max(1, Math.round(image.naturalWidth * scale));
    canvas.height = Math.max(1, Math.round(image.naturalHeight * scale));
    const context = canvas.getContext("2d");

    if (!context) {
      throw new Error("画像を処理するCanvasを作成できませんでした。");
    }

    context.drawImage(image, 0, 0, canvas.width, canvas.height);
    return canvas;
  } finally {
    URL.revokeObjectURL(url);
  }
}

function candidateLabel(result) {
  const candidate = CANDIDATES.find((item) => item.prompt === result.label);
  return candidate?.label ?? result.label;
}

function showSuggestions(results, areaResults) {
  suggestions.replaceChildren();
  areas.replaceChildren();

  for (const result of results.slice(0, 3)) {
    const item = document.createElement("li");
    item.textContent = candidateLabel(result);
    suggestions.append(item);
  }

  for (const { name, results: areaMatches } of areaResults) {
    const item = document.createElement("li");
    const bestMatch = areaMatches[0];
    const description = candidateLabel(bestMatch);
    const resultText = description.startsWith("明確な兆候")
      ? "他の候補との比較で、明確な兆候は上位になりませんでした"
      : `確認候補: ${description}`;
    item.textContent = `${name}: ${resultText}`;
    areas.append(item);
  }
}

function cropCanvas(source, left, top, width, height) {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext("2d");

  if (!context) {
    throw new Error("画像範囲を作成するCanvasを用意できませんでした。");
  }

  context.drawImage(source, left, top, width, height, 0, 0, width, height);
  return canvas;
}

function getImageAreas(source) {
  const middleX = Math.floor(source.width / 2);
  const middleY = Math.floor(source.height / 2);
  const rightWidth = source.width - middleX;
  const bottomHeight = source.height - middleY;

  return [
    { name: "左上", canvas: cropCanvas(source, 0, 0, middleX, middleY) },
    { name: "右上", canvas: cropCanvas(source, middleX, 0, rightWidth, middleY) },
    { name: "左下", canvas: cropCanvas(source, 0, middleY, middleX, bottomHeight) },
    { name: "右下", canvas: cropCanvas(source, middleX, middleY, rightWidth, bottomHeight) },
  ];
}

photoInput.addEventListener("change", () => {
  const file = photoInput.files?.[0];
  suggestions.replaceChildren();
  observationInputs.forEach((input) => {
    input.checked = false;
  });
  updateRiskLevel();

  if (!file) {
    analyzeButton.disabled = true;
    photoName.textContent = "写真が選択されていません。";
    preview.hidden = true;
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = undefined;
    return;
  }

  if (!file.type.startsWith("image/")) {
    photoInput.value = "";
    analyzeButton.disabled = true;
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = undefined;
    preview.removeAttribute("src");
    preview.hidden = true;
    photoName.textContent = "写真が選択されていません。";
    status.textContent = "画像ファイルを選択してください。";
    return;
  }

  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = URL.createObjectURL(file);
  preview.src = previewUrl;
  preview.hidden = false;
  photoName.textContent = file.name;
  analyzeButton.disabled = false;
  status.textContent = "写真を確認できます。";
});

analyzeButton.addEventListener("click", async () => {
  if (busy) return;
  const file = photoInput.files?.[0];
  if (!file) {
    status.textContent = "先に写真を選択してください。";
    return;
  }

  setBusy(true);
  suggestions.replaceChildren();
  status.textContent = "WebGPUを確認しています。";

  try {
    await requireWebGPU();
    const canvas = await makeInputCanvas(file);
    const image = RawImage.fromCanvas(canvas);
    const model = await getClassifier();
    const prompts = CANDIDATES.map((item) => item.prompt);
    status.textContent = "写真全体を確認中です。";
    const results = await model(image, prompts);
    const areaResults = [];

    for (const area of getImageAreas(canvas)) {
      status.textContent = `${area.name}の範囲を確認中です。`;
      const areaImage = RawImage.fromCanvas(area.canvas);
      const matches = await model(areaImage, prompts);
      areaResults.push({ name: area.name, results: matches });
    }

    showSuggestions(results, areaResults);
    status.textContent = "候補を表示しました。現場で確認できた兆候を選んでください。";
  } catch (error) {
    console.error(error);
    status.textContent = `確認に失敗しました: ${error.message}`;
  } finally {
    setBusy(false);
  }
});

observationInputs.forEach((input) => {
  input.addEventListener("change", updateRiskLevel);
});

window.addEventListener("pagehide", () => {
  classifier?.dispose?.();
  if (previewUrl) URL.revokeObjectURL(previewUrl);
});
