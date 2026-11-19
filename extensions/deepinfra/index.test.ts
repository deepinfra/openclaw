import type { Model } from "openclaw/plugin-sdk/llm";
import {
  createCapturedPluginRegistration,
  registerSingleProviderPlugin,
} from "openclaw/plugin-sdk/plugin-test-runtime";
import { clearLiveCatalogCacheForTests } from "openclaw/plugin-sdk/provider-catalog-live-runtime";
import type { ProviderCatalogContext } from "openclaw/plugin-sdk/provider-catalog-shared";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { captureProviderRequestPayload } from "../test-support/provider-request-payload.test-support.js";
import deepinfraPlugin from "./index.js";
import { DEEPINFRA_MODEL_CATALOG } from "./provider-static-catalog.js";

const baseUrl = "https://api.deepinfra.com/v1/openai";
const DEEPINFRA_MODELS_URL = `${baseUrl}/models?sort_by=openclaw&filter=with_meta`;
const providerConfig = { baseUrl, api: "openai-completions", apiKey: "profile-key" };
const cost = { input: 4, output: 8, cacheRead: 0, cacheWrite: 0 };
const liveModel = expect.objectContaining({ id: "profile/live-model", cost });

const catalogContext: ProviderCatalogContext = {
  config: {},
  env: {},
  agentDir: "/tmp/openclaw-agent",
  resolveProviderApiKey: () => ({ apiKey: "profile-key" }),
  resolveProviderAuth: () => ({ apiKey: "profile-key", mode: "api_key", source: "profile" }),
};

async function discoveryResponse(url: string) {
  const id = "profile/live-model";
  if (url === DEEPINFRA_MODELS_URL) {
    return Response.json({ data: [{ id, metadata: { tags: ["chat"] } }] });
  }
  expect(url).toBe("https://api.deepinfra.com/models/list");
  return Response.json([
    {
      model_name: id,
      pricing: {
        type: "tokens",
        cents_per_input_token: 0.0004,
        cents_per_output_token: 0.0008,
      },
    },
  ]);
}

afterEach(() => {
  clearLiveCatalogCacheForTests();
  vi.unstubAllEnvs();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

describe("deepinfra capability registration", () => {
  it.each([
    { baseUrl, expected: "synthetic-session" },
    { baseUrl: `${baseUrl}/`, expected: "synthetic-session" },
    { baseUrl: "https://proxy.example/v1", expected: undefined },
    { baseUrl: "https://api.deepinfra.com/custom/v1", expected: undefined },
    { baseUrl, optOut: true, expected: undefined },
    { baseUrl, disabled: true, expected: undefined },
  ])("sends native cache affinity: $baseUrl, optOut=$optOut, disabled=$disabled", async (route) => {
    const provider = await registerSingleProviderPlugin(deepinfraPlugin);
    const model: Model = {
      id: "synthetic-model",
      name: "Synthetic model",
      provider: "deepinfra",
      api: "openai-completions",
      reasoning: false,
      input: ["text"],
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
      contextWindow: 8192,
      maxTokens: 128,
      baseUrl: route.baseUrl,
      ...(route.optOut ? { compat: { supportsPromptCacheKey: false } } : {}),
    };
    const normalized =
      provider.normalizeResolvedModel?.({ provider: model.provider, modelId: model.id, model }) ??
      model;
    const { result, payload } = await captureProviderRequestPayload(normalized, {
      cacheRetention: route.disabled ? "none" : "long",
    });
    expect(result.errorMessage).toBe("captured before request");
    expect(payload).toMatchObject({ prompt_cache_key: route.expected });
    expect(payload).not.toHaveProperty("prompt_cache_retention");
    expect(payload).not.toHaveProperty("prompt_cache_options");
  });

  it.each([
    ...["metadata", "pricing"].flatMap((scenario) =>
      [401, 503].map((status) => ({ scenario, status })),
    ),
    { scenario: "empty", status: 503 },
    { scenario: "healthy", status: 200 },
  ])("handles $scenario catalog discovery with HTTP $status", async ({ scenario, status }) => {
    const mockFetch = vi.fn(async (url: string) => {
      const metadata = url === DEEPINFRA_MODELS_URL;
      if (scenario === "empty" && metadata) {
        return Response.json({ data: [] });
      }
      if (
        (scenario === "metadata" && metadata) ||
        ((scenario === "pricing" || scenario === "empty") && !metadata)
      ) {
        return new Response("unavailable", { status });
      }
      return discoveryResponse(url);
    });
    const provider = await registerSingleProviderPlugin(deepinfraPlugin);
    vi.stubGlobal("fetch", mockFetch);
    const result = await provider.catalog?.run(catalogContext);
    const rejected = status === 401 || status === 403;
    const ready = scenario === "empty" || scenario === "healthy";
    const catalog = {
      ...providerConfig,
      models: scenario === "empty" ? [] : [liveModel],
    };
    const expected = ready
      ? { provider: scenario === "empty" ? catalog : expect.objectContaining(catalog) }
      : { providers: {} };
    expect(result).toEqual({
      ...expected,
      outcomes: [
        {
          provider: "deepinfra",
          status: ready ? "ready" : rejected ? "auth-rejected" : "unavailable",
          ...(rejected ? { rejectionScope: "catalog" } : {}),
        },
      ],
    });
  });

  it("registers all DeepInfra-backed OpenClaw provider surfaces", () => {
    const captured = createCapturedPluginRegistration();
    deepinfraPlugin.register(captured.api);

    for (const providers of [
      captured.providers,
      captured.imageGenerationProviders,
      captured.mediaUnderstandingProviders,
      captured.embeddingProviders,
      captured.speechProviders,
      captured.videoGenerationProviders,
    ]) {
      expect(providers.map((provider) => provider.id)).toEqual(["deepinfra"]);
    }
  });

  it.each([
    { modelIds: ["Anthropic/Claude-4-Sonnet", "ANTHROPIC/claude-4-sonnet"], eligible: true },
    { modelIds: ["meta-llama/Llama-4-Scout-17B-16E-Instruct", "zai-org/GLM-5.1"], eligible: false },
  ])("classifies cache TTL eligibility as $eligible", async ({ modelIds, eligible }) => {
    const provider = await registerSingleProviderPlugin(deepinfraPlugin);
    for (const modelId of modelIds) {
      expect(provider.isCacheTtlEligible?.({ provider: "deepinfra", modelId })).toBe(eligible);
    }
  });
});

// Regression: live-catalog ids offered by the picker/models.list must also
// resolve for agent runs; before the dynamic hooks existed every non-static
// model failed with "Unknown model" (picker offered 142, only ~12 ran).
describe("deepinfra dynamic model resolution", () => {
  // No key => discovery serves the manifest; keeps a developer's real key out of the run.
  beforeEach(() => {
    vi.stubEnv("DEEPINFRA_API_KEY", "");
  });

  it("prepares discovered catalog models for async run resolution", async () => {
    const provider = await registerSingleProviderPlugin(deepinfraPlugin);
    const manifestModel = DEEPINFRA_MODEL_CATALOG[0]!;
    const context = {
      provider: "deepinfra",
      modelId: manifestModel.id,
      agentDir: "/tmp/openclaw-agent",
      modelRegistry: { find: vi.fn(() => null) },
    } as never;

    const model = await provider.prepareDynamicModel?.(context);

    expect(model).toMatchObject({
      id: manifestModel.id,
      provider: "deepinfra",
      api: "openai-completions",
      baseUrl: "https://api.deepinfra.com/v1/openai",
    });
  });

  it("resolves ids missing from discovery to a default-capability model instead of dead-ending", async () => {
    const provider = await registerSingleProviderPlugin(deepinfraPlugin);
    const context = {
      provider: "deepinfra",
      modelId: "synthetic/Fresh-Model-Next",
      agentDir: "/tmp/openclaw-agent",
      modelRegistry: { find: vi.fn(() => null) },
    } as never;

    expect(await provider.prepareDynamicModel?.(context)).toBeUndefined();
    const model = provider.resolveDynamicModel?.(context);

    expect(model).toMatchObject({
      id: "synthetic/Fresh-Model-Next",
      provider: "deepinfra",
      api: "openai-completions",
      baseUrl: "https://api.deepinfra.com/v1/openai",
      input: ["text"],
      contextWindow: 128000,
      maxTokens: 8192,
    });
  });

  it("keeps the deepseek thinking format on dynamic deepseek-ai models", async () => {
    const provider = await registerSingleProviderPlugin(deepinfraPlugin);
    const model = provider.resolveDynamicModel?.({
      provider: "deepinfra",
      modelId: "deepseek-ai/Synthetic-Reasoner",
      agentDir: "/tmp/openclaw-agent",
      modelRegistry: { find: vi.fn(() => null) },
    } as never);

    expect(model?.compat).toMatchObject({ thinkingFormat: "deepseek" });
  });

  it("honors a configured provider baseUrl for dynamic models", async () => {
    const provider = await registerSingleProviderPlugin(deepinfraPlugin);
    const model = provider.resolveDynamicModel?.({
      provider: "deepinfra",
      modelId: "synthetic/model",
      agentDir: "/tmp/openclaw-agent",
      modelRegistry: { find: vi.fn(() => null) },
      providerConfig: { baseUrl: "https://proxy.example.invalid/v1/openai" },
    } as never);

    expect(model?.baseUrl).toBe("https://proxy.example.invalid/v1/openai");
  });

  it("serves discovered metadata to sync resolution after a prepare pass", async () => {
    const provider = await registerSingleProviderPlugin(deepinfraPlugin);
    const manifestModel = DEEPINFRA_MODEL_CATALOG[0]!;
    const context = {
      provider: "deepinfra",
      modelId: manifestModel.id,
      agentDir: "/tmp/openclaw-agent",
      modelRegistry: { find: vi.fn(() => null) },
    } as never;

    await provider.prepareDynamicModel?.(context);
    const model = provider.resolveDynamicModel?.(context);

    expect(model).toMatchObject({
      id: manifestModel.id,
      name: manifestModel.name ?? manifestModel.id,
    });
    if (typeof manifestModel.contextWindow === "number") {
      expect(model?.contextWindow).toBe(manifestModel.contextWindow);
    }
  });
});
