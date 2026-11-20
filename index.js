/**
 * Trace every DeepSeek Harness conversation turn into MLflow.
 *
 * The Session log is the harness's source of truth: every model call, tool
 * invocation, and turn boundary is committed there before anything else sees
 * it. This plugin subscribes to the post-commit `session/event` feed and
 * projects that log into one MLflow trace per turn — an AGENT root span for the
 * turn, a CHAIN span per step, an LLM span per model call, and a TOOL span per
 * tool invocation — then hands the trace to the MLflow TypeScript SDK, which
 * exports to the workspace's MLflow experiment.
 *
 * Credentials come from the configured Databricks profile and are passed to that
 * SDK as explicit values: the CLI is asked for a token for the named profile,
 * and the host is read from the same configuration file. Using the SDK's own
 * profile lookup instead would shell out to `databricks auth token --host …`,
 * which the CLI refuses when two profiles name one workspace.
 *
 * Tracing follows the log rather than the live event stream on purpose: a
 * cancelled turn still closes its spans, a retried model attempt is still
 * visible, and the trace never contains anything the session did not commit.
 *
 * @module @local/dsh-databricks-mlflow-tracing
 */

import { execFile } from 'node:child_process'
import { readFile } from 'node:fs/promises'
import { homedir, userInfo } from 'node:os'
import { join } from 'node:path'

/**
 * Registered plugin name, also the prefix of every log line.
 * @type {string}
 */
export const name = 'dsh-databricks-mlflow-tracing'

/**
 * No service is required: the plugin only listens to the post-commit
 * `session/event` feed, which needs nothing but a mounting context.
 * @type {string[]}
 */
export const inject = []

/**
 * Defaults overridden by the row's `config`.
 *
 * `profile` is empty on purpose: an empty value lets the Databricks SDK pick
 * its own profile (`DATABRICKS_CONFIG_PROFILE`, else the `DEFAULT` entry in
 * `~/.databrickscfg`), so the bundle ships no machine-specific value.
 */
const DEFAULTS = {
  /** Databricks CLI/config profile whose workspace receives the traces. */
  profile: '',
  /** Explicit MLflow tracking URI; empty derives the Databricks workspace from the profile. */
  trackingUri: '',
  /** Experiment name; a relative name lands under `/Users/<you>/` on Databricks. */
  experiment: 'dsh-conversations',
  /** Experiment ID. When set, no experiment lookup happens at all. */
  experimentId: '',
  /** Alternate `~/.databrickscfg` location used to read the profile's host. */
  databricksConfigPath: '',
  /** Absolute path to the Databricks CLI; empty searches `PATH` and common install locations. */
  cliPath: '',
  /** Bearer token for an OSS MLflow server; empty falls back to the environment. */
  trackingToken: '',
  /** Capture prompts, model output, reasoning, and tool payloads. */
  captureContent: true,
  /** Include assistant reasoning content in the LLM span. */
  includeReasoning: true,
  /** Maximum characters kept for one prompt, response, or reasoning field. */
  maxChars: 20_000,
  /** Maximum characters kept for one tool result. */
  maxToolResultChars: 4_000,
  /** Root span name prefix; the turn number is appended. */
  traceName: 'dsh turn',
  /** Replace the OAuth access token once it expires within this window. */
  refreshSkewMs: 10 * 60 * 1000,
  /** Kill a Databricks CLI call after this long. */
  timeoutMs: 20 * 1000,
  /** How long a failed initialization waits before the next attempt. */
  retryIntervalMs: 5 * 60 * 1000,
  /** Extra trace tags, applied to every trace. */
  tags: {},
}

/**
 * Prefer the Cordis logger, which writes to the application log, and fall back
 * to the console when the plugin is mounted without one.
 * @param {object} ctx - the plugin context.
 * @returns {object} something with `info` and `warn` methods.
 */
function pickLogger(ctx) {
  try {
    if (typeof ctx.logger === 'function') {
      const logger = ctx.logger(name)
      if (logger !== undefined && logger !== null) return logger
    }
  } catch {
    // A composition without a logger still gets its warnings on stdout.
  }
  return console
}

/**
 * The most useful explanation an error carries.
 * @param {unknown} error - anything thrown.
 * @returns {string} a one-paragraph explanation.
 */
function describe(error) {
  if (error instanceof Error) return error.message === '' ? error.name : error.message
  if (typeof error === 'string') return error
  try {
    return JSON.stringify(error)
  } catch {
    return String(error)
  }
}

/**
 * Read one configuration value as a trimmed string.
 * @param {unknown} value - the configured value.
 * @param {string} fallback - the value used when nothing usable was supplied.
 * @returns {string} the normalized string.
 */
function stringOption(value, fallback) {
  if (typeof value === 'string' && value.trim() !== '') return value.trim()
  if (typeof value === 'number' && Number.isFinite(value)) return String(value)
  return fallback
}

/**
 * Read one configuration value as a boolean.
 * @param {unknown} value - the configured value.
 * @param {boolean} fallback - the value used when nothing usable was supplied.
 * @returns {boolean} the normalized boolean.
 */
function booleanOption(value, fallback) {
  return typeof value === 'boolean' ? value : fallback
}

/**
 * Read one configuration value as a positive number.
 * @param {unknown} value - the configured value.
 * @param {number} fallback - the value used when nothing usable was supplied.
 * @param {number} minimum - the smallest accepted value.
 * @returns {number} the normalized number.
 */
function numberOption(value, fallback, minimum) {
  const parsed = Number(value)
  return Number.isFinite(parsed) && parsed >= minimum ? parsed : fallback
}

/**
 * Merge the row's config over the defaults, normalizing every field.
 * @param {object | undefined} config - the row's config.
 * @returns {typeof DEFAULTS} complete, validated options.
 */
function normalizeConfig(config) {
  const input = config ?? {}
  const tags = {}
  if (input.tags !== null && typeof input.tags === 'object') {
    for (const [key, value] of Object.entries(input.tags)) {
      if (value !== undefined && value !== null) tags[String(key)] = String(value)
    }
  }
  return {
    profile: stringOption(input.profile, DEFAULTS.profile),
    trackingUri: stringOption(input.trackingUri, DEFAULTS.trackingUri),
    experiment: stringOption(input.experiment, DEFAULTS.experiment),
    experimentId: stringOption(input.experimentId, DEFAULTS.experimentId),
    databricksConfigPath: stringOption(input.databricksConfigPath, DEFAULTS.databricksConfigPath),
    cliPath: stringOption(input.cliPath, DEFAULTS.cliPath),
    trackingToken: stringOption(input.trackingToken, DEFAULTS.trackingToken),
    captureContent: booleanOption(input.captureContent, DEFAULTS.captureContent),
    includeReasoning: booleanOption(input.includeReasoning, DEFAULTS.includeReasoning),
    maxChars: numberOption(input.maxChars, DEFAULTS.maxChars, 200),
    maxToolResultChars: numberOption(input.maxToolResultChars, DEFAULTS.maxToolResultChars, 200),
    traceName: stringOption(input.traceName, DEFAULTS.traceName),
    refreshSkewMs: numberOption(input.refreshSkewMs, DEFAULTS.refreshSkewMs, 0),
    timeoutMs: numberOption(input.timeoutMs, DEFAULTS.timeoutMs, 1000),
    retryIntervalMs: numberOption(input.retryIntervalMs, DEFAULTS.retryIntervalMs, 1000),
    tags,
  }
}

/**
 * Clip one captured string to the configured budget.
 *
 * A trace is a debugging artifact, not a transcript: an 800 KB tool result
 * makes the trace unreadable in the UI and pushes the trace artifact over the
 * backend's size limits. Truncation is announced in the value so a reader can
 * tell the difference between "the tool printed little" and "we kept little".
 * @param {string} text - the captured text.
 * @param {number} limit - maximum characters to keep.
 * @returns {string} the clipped text.
 */
function clip(text, limit) {
  if (typeof text !== 'string') return ''
  if (text.length <= limit) return text
  return `${text.slice(0, limit)}\n… [truncated ${text.length - limit} of ${text.length} characters]`
}

/**
 * Normalize one span-attribute map.
 *
 * The MLflow SDK serializes every attribute through OpenTelemetry, where a
 * numeric `0`, an empty string, and `false` are indistinguishable from an
 * absent value by the time the backend reads the span. Sending the string form
 * of a primitive keeps a genuine zero distinguishable from a missing field in
 * the trace UI.
 * @param {Record<string, unknown>} attributes - the attributes to set.
 * @returns {Record<string, unknown>} attributes safe to hand to the SDK.
 */
function spanAttributes(attributes) {
  const normalized = {}
  for (const [key, value] of Object.entries(attributes)) {
    if (value === undefined || value === null) continue
    if (typeof value === 'number' || typeof value === 'boolean') normalized[key] = String(value)
    else if (typeof value === 'string' && value === '') continue
    else normalized[key] = value
  }
  return normalized
}

/**
 * The MLflow SDK's span-status values, spelled as OTLP names them.
 *
 * The SDK compares against these `STATUS_CODE_*` spellings, not the short
 * `OK`/`ERROR` names, and silently downgrades anything else to UNSET — which
 * the span's own `end()` then rewrites to OK. A failed turn or tool call would
 * otherwise be reported as a success.
 * @type {Readonly<Record<'OK' | 'ERROR' | 'UNSET', string>>}
 */
const SPAN_STATUS = Object.freeze({
  OK: 'STATUS_CODE_OK',
  ERROR: 'STATUS_CODE_ERROR',
  UNSET: 'STATUS_CODE_UNSET',
})

/**
 * The SDK's span status value for a status kind.
 * @param {'OK' | 'ERROR' | 'UNSET'} kind - the status kind.
 * @returns {string} the value the SDK accepts.
 */
function spanStatus(kind) {
  return SPAN_STATUS[kind] ?? SPAN_STATUS.UNSET
}

/**
 * Convert an event timestamp to the nanosecond value the MLflow SDK wants.
 * @param {unknown} milliseconds - epoch milliseconds.
 * @returns {number | undefined} nanoseconds, or undefined when unusable.
 */
function toNanoseconds(milliseconds) {
  return typeof milliseconds === 'number' && Number.isFinite(milliseconds) && milliseconds > 0
    ? Math.round(milliseconds * 1e6)
    : undefined
}

/**
 * The earliest model-stream timestamp, used as the LLM span start.
 *
 * `assistant/message` is committed when the model call has already finished, so
 * its event time is the end of the span. The embedded compact stream keeps the
 * original timing evidence, which is what lets the span cover the real call
 * rather than only the commit. Records carry either `time` (one chunk) or
 * `time0` plus deltas (a run of chunks).
 * @param {unknown} stream - the event's compact stream records.
 * @param {number | undefined} fallbackNs - start to use when no timing exists.
 * @returns {number | undefined} the nanoseconds of the first recorded frame.
 */
function streamStartNs(stream, fallbackNs) {
  if (!Array.isArray(stream)) return fallbackNs
  let earliest = Number.POSITIVE_INFINITY
  for (const record of stream) {
    if (record === null || typeof record !== 'object') continue
    if (typeof record.time === 'number' && record.time < earliest) earliest = record.time
    if (typeof record.time0 === 'number' && record.time0 < earliest) earliest = record.time0
  }
  return Number.isFinite(earliest) ? toNanoseconds(earliest) : fallbackNs
}

/**
 * One content block as plain text, or an empty string for non-text blocks.
 * @param {unknown} block - a model or tool content block.
 * @returns {string} the block's text.
 */
function blockText(block) {
  if (block === null || typeof block !== 'object') return ''
  switch (block.type) {
    case 'text':
    case 'reasoning':
      return typeof block.text === 'string' ? block.text : ''
    case 'image':
      return '[image]'
    case 'file':
      return `[file ${typeof block.attachment?.name === 'string' ? block.attachment.name : 'attachment'}]`
    case 'tool-call':
      return ''
    default:
      return ''
  }
}

/**
 * Split a content array into the parts a span records.
 * @param {unknown} content - the message content array.
 * @param {boolean} includeReasoning - keep reasoning text.
 * @returns {{text: string, reasoning: string, toolCalls: object[]}} the parts.
 */
function readContent(content, includeReasoning) {
  const text = []
  const reasoning = []
  const toolCalls = []
  if (!Array.isArray(content)) return { text: '', reasoning: '', toolCalls }
  for (const block of content) {
    if (block === null || typeof block !== 'object') continue
    if (block.type === 'tool-call') {
      toolCalls.push({ name: typeof block.name === 'string' ? block.name : 'tool', arguments: block.arguments })
    } else if (block.type === 'reasoning') {
      if (includeReasoning) reasoning.push(blockText(block))
    } else {
      const value = blockText(block)
      if (value !== '') text.push(value)
    }
  }
  return { text: text.join('\n'), reasoning: reasoning.join('\n'), toolCalls }
}

/**
 * Parse a tool's raw argument JSON, falling back to the raw string.
 * @param {unknown} raw - the argument string exactly as the model produced it.
 * @param {number} limit - maximum characters kept.
 * @returns {unknown} a JSON value, or the clipped raw string.
 */
function parseArguments(raw, limit) {
  if (typeof raw !== 'string') return raw
  if (raw.length > limit) return clip(raw, limit)
  try {
    return JSON.parse(raw)
  } catch {
    return raw
  }
}

/**
 * Whether a tracking URI names a Databricks workspace rather than an MLflow
 * server reached over plain HTTP.
 * @param {string} trackingUri - the resolved tracking URI.
 * @returns {boolean} true for `databricks` and `databricks://…` forms.
 */
function isDatabricksTracking(trackingUri) {
  return trackingUri === 'databricks' || trackingUri.startsWith('databricks://')
}

/**
 * Where the Databricks CLI is usually installed on a GUI-launched macOS app,
 * whose `PATH` is the launcher's minimal one rather than the login shell's.
 * @type {readonly string[]}
 */
const CLI_FALLBACKS = Object.freeze([
  '/opt/homebrew/bin/databricks',
  '/usr/local/bin/databricks',
  join(homedir(), '.local/bin/databricks'),
])

/**
 * The most useful explanation a failed CLI run produced.
 *
 * The Databricks CLI reports many failures as a JSON object on stdout with a
 * non-zero exit and an empty stderr, so a stderr-only message would discard the
 * one line that says what to do. Stderr wins when it has content, then the
 * CLI's own JSON message, then the process message.
 * @param {unknown} error - the `execFile` error.
 * @param {string | Buffer | undefined} stdout - captured standard output.
 * @param {string | Buffer | undefined} stderr - captured standard error.
 * @returns {string} the failure detail.
 */
function failureDetail(error, stdout, stderr) {
  const stderrText = String(stderr ?? '').trim()
  if (stderrText !== '') return stderrText
  const stdoutText = String(stdout ?? '').trim()
  if (stdoutText.startsWith('{')) {
    try {
      const parsed = JSON.parse(stdoutText)
      if (typeof parsed?.message === 'string' && parsed.message !== '') return parsed.message
    } catch {
      // Not the CLI's error envelope; fall through to the process message.
    }
  }
  return error instanceof Error ? error.message : String(error)
}

/**
 * Run one executable and resolve its stdout.
 * @param {string} file - executable path or a bare name resolved through `PATH`.
 * @param {string[]} args - command arguments.
 * @param {number} timeoutMs - kill the child after this long.
 * @returns {Promise<string>} standard output.
 */
function execOnce(file, args, timeoutMs) {
  return new Promise((resolve, reject) => {
    execFile(file, args, { timeout: timeoutMs, maxBuffer: 8 * 1024 * 1024, windowsHide: true }, (error, stdout, stderr) => {
      if (error) {
        const failure = new Error(failureDetail(error, stdout, stderr))
        failure.code = error.code
        reject(failure)
        return
      }
      resolve(stdout)
    })
  })
}

/**
 * Run the Databricks CLI, honoring the configured path and falling back to
 * `PATH` and the usual install locations.
 *
 * A GUI-launched Desktop app inherits a minimal `PATH`, so a bare `databricks`
 * is not reliably resolvable; the fallbacks make an unconfigured plugin work on
 * a normal macOS install, and failing all of them names the setting to fix
 * rather than the bare `spawn ENOENT` a reader would have to decode.
 * @param {string} cliPath - configured executable path, possibly empty.
 * @param {string[]} args - command arguments.
 * @param {number} timeoutMs - kill the child after this long.
 * @returns {Promise<string>} standard output.
 */
async function runCli(cliPath, args, timeoutMs) {
  const candidates = cliPath === '' ? ['databricks', ...CLI_FALLBACKS] : [cliPath, 'databricks', ...CLI_FALLBACKS]
  for (const candidate of candidates) {
    try {
      return await execOnce(candidate, args, timeoutMs)
    } catch (error) {
      if (error.code !== 'ENOENT') throw error
    }
  }
  throw new Error(`Databricks CLI not found (tried ${candidates.join(', ')}); set cliPath to its absolute path`)
}

/**
 * Read the access token and its deadline out of `databricks auth token -o json`.
 * @param {string} raw - the CLI's standard output.
 * @returns {{token: string, expiresAt: number}} the token and its deadline.
 */
function readCliToken(raw) {
  let parsed
  try {
    parsed = JSON.parse(raw)
  } catch {
    throw new Error('`databricks auth token` did not print JSON')
  }
  const token = parsed?.access_token
  if (typeof token !== 'string' || token.length === 0) {
    throw new Error('`databricks auth token` printed no access_token')
  }
  const expiry = typeof parsed.expiry === 'string' ? Date.parse(parsed.expiry) : Number.NaN
  const expiresAt = Number.isFinite(expiry)
    ? expiry
    : Number(parsed.expires_in) > 0
      ? Date.now() + Number(parsed.expires_in) * 1000
      : Date.now() + 30 * 60 * 1000
  return { token, expiresAt }
}

/**
 * Read one profile's `host` out of a Databricks configuration file.
 *
 * The host is needed as an explicit config value: the Databricks SDK's own
 * profile auth shells out to `databricks auth token --host <host>`, which the
 * CLI refuses when two profiles name the same workspace. Passing an explicit
 * profile to the CLI and an explicit host and token to the SDK removes that
 * ambiguity entirely.
 * @param {string} configPath - configured file path, or empty for `~/.databrickscfg`.
 * @param {string} profile - the profile section to read.
 * @returns {Promise<string | undefined>} the host, without a trailing slash.
 */
async function readProfileHost(configPath, profile) {
  const file = configPath !== '' ? configPath : join(homedir(), '.databrickscfg')
  let text
  try {
    text = await readFile(file, 'utf8')
  } catch {
    return undefined
  }
  let section
  for (const line of text.split('\n')) {
    const trimmed = line.trim()
    if (trimmed === '' || trimmed.startsWith('#') || trimmed.startsWith(';')) continue
    const header = /^\[(.+)\]$/.exec(trimmed)
    if (header !== null) {
      section = header[1].trim()
      continue
    }
    if (section !== profile) continue
    const match = /^host\s*=\s*(.+)$/.exec(trimmed)
    if (match !== null) return (match[1] ?? '').trim().replace(/\/+$/, '')
  }
  return undefined
}

/**
 * Trace every turn of every session through one MLflow experiment.
 *
 * Activation resolves the tracking target before the plugin starts listening,
 * so the first turn of the first session is traced. A failed resolution is
 * logged rather than thrown: rejecting activation would leave the user with no
 * plugin to reconfigure and no trace to diagnose.
 * @param {object} ctx - the plugin context.
 * @param {object | undefined} config - the row's config, overriding {@link DEFAULTS}.
 */
export async function apply(ctx, config) {
  const options = normalizeConfig(config)
  const log = pickLogger(ctx)
  const label = `databricks-mlflow-tracing[${options.profile === '' ? 'default' : options.profile}]`

  /** @type {{mlflow: any, otel: any} | undefined} */
  let sdk
  /** @type {{trackingUri: string, experimentId: string, host?: string, token?: string} | undefined} */
  let target
  /** @type {{profile: string, host: string, token: string, expiresAt: number} | undefined} */
  let credentials
  let ready = false
  let lastError = ''
  let lastAttempt = 0
  let lastRefreshAttempt = 0
  let stopped = false

  /** @type {Map<string, object>} */
  const sessions = new Map()

  let hostUser = 'unknown'
  try {
    hostUser = userInfo().username
  } catch {
    // A container without a passwd entry still traces; the field stays generic.
  }

  // --- Target resolution ---------------------------------------------------

  /**
   * Load the MLflow tracing SDK and, when present, the OpenTelemetry API used
   * to attach trace-level metadata.
   *
   * Both are loaded dynamically so a missing dependency produces one actionable
   * warning on the plugin row instead of a module-load failure the Loader would
   * report without naming the plugin.
   * @returns {Promise<{mlflow: any, otel: any}>} the loaded modules.
   */
  /**
   * One SDK per process, shared across plugin reloads.
   *
   * `initializeSDK()` inside the MLflow SDK shuts the previous SDK down without
   * awaiting it and then starts a new one. In a host process that has already
   * registered an OpenTelemetry provider, the replacement never receives spans:
   * the old provider is gone, the new one is unattached, and `flushTraces()`
   * resolves in a millisecond having exported nothing. Reusing the instance
   * that is already exporting avoids that entirely.
   */
  async function loadSdk() {
    const key = Symbol.for('dsh.databricks-mlflow-tracing.sdk')
    const cached = globalThis[key]
    if (cached !== undefined) return cached
    const imported = await import('mlflow-tracing')
    const mlflow = typeof imported?.init === 'function' ? imported : imported?.default
    if (mlflow === undefined || typeof mlflow.init !== 'function') {
      throw new Error('the mlflow-tracing package did not export init()')
    }
    let otel
    try {
      otel = await import('@opentelemetry/api')
    } catch {
      otel = undefined
    }
    const built = { mlflow, otel }
    globalThis[Symbol.for('dsh.databricks-mlflow-tracing.sdk')] = built
    return built
  }

  /**
   * The profile that supplies the workspace and its credential.
   *
   * An empty configuration follows `DATABRICKS_CONFIG_PROFILE`, else the
   * `DEFAULT` section, which is also what the CLI itself would use.
   * @returns {string} the profile name.
   */
  function profileName() {
    if (options.profile !== '') return options.profile
    const fromEnvironment = (process.env.DATABRICKS_CONFIG_PROFILE ?? '').trim()
    return fromEnvironment !== '' ? fromEnvironment : 'DEFAULT'
  }

  /**
   * Read the workspace host and a fresh OAuth access token for one profile.
   *
   * Both halves are explicit: the CLI is asked for the named profile, so two
   * profiles naming one workspace cannot make it refuse, and the host comes
   * from the same file rather than from a host-based lookup.
   * @param {string} profile - the profile to read.
   * @param {boolean} force - bypass the CLI's cached, still-valid token.
   * @returns {Promise<{profile: string, host: string, token: string, expiresAt: number}>} the credentials.
   */
  async function resolveCredentials(profile, force) {
    const fromEnvironment = (process.env.DATABRICKS_HOST ?? '').trim().replace(/\/+$/, '')
    const host = fromEnvironment !== ''
      ? fromEnvironment
      : await readProfileHost(options.databricksConfigPath, profile)
    if (host === undefined || host === '') {
      throw new Error(`profile "${profile}" has no host in ~/.databrickscfg; run databricks auth login --profile ${profile}, or set DATABRICKS_HOST`)
    }
    const args = ['auth', 'token', '-o', 'json', '-p', profile]
    if (force) args.push('--force-refresh')
    let raw
    try {
      raw = await runCli(options.cliPath, args, options.timeoutMs)
    } catch (error) {
      const detail = describe(error)
      throw new Error(/refresh token|reauthenticate|login/i.test(detail)
        ? `profile "${profile}" cannot mint a token: ${detail}`
        : `cannot read a Databricks token for profile "${profile}": ${detail}`)
    }
    const { token, expiresAt } = readCliToken(raw)
    return { profile, host, token, expiresAt }
  }

  /**
   * Resolve the experiment ID on a Databricks workspace through the Databricks
   * SDK, authenticated by the profile's own token.
   *
   * The experiment name defaults to the current user's home folder so a fresh
   * workspace needs no configuration: `/Users/<you>/dsh-conversations`.
   * @param {string} name - the configured experiment name.
   * @param {{host: string, token: string}} auth - the resolved credentials.
   * @returns {Promise<string>} the experiment ID.
   */
  async function resolveDatabricksExperiment(name, auth) {
    const databricks = await import('@databricks/sdk-experimental')
    const client = new databricks.WorkspaceClient({ host: auth.host, token: auth.token })
    const me = await client.currentUser.me()
    const userName = typeof me?.userName === 'string' && me.userName !== '' ? me.userName : undefined
    if (userName === undefined) {
      throw new Error('the Databricks workspace returned no user name, so a relative experiment name cannot be placed')
    }
    const experimentName = name.startsWith('/') ? name : `/Users/${userName}/${name}`
    try {
      const found = await client.experiments.getByName({ experiment_name: experimentName })
      const id = found?.experiment?.experiment_id
      if (id !== undefined && id !== null && String(id) !== '') return String(id)
    } catch {
      // A missing experiment answers with an error; creation below owns the outcome.
    }
    try {
      const created = await client.experiments.createExperiment({ name: experimentName })
      const id = created?.experiment_id
      if (id !== undefined && id !== null && String(id) !== '') {
        log.info?.(`${label}: created experiment ${experimentName} (${id})`)
        return String(id)
      }
    } catch (error) {
      throw new Error(`cannot read or create experiment "${experimentName}": ${describe(error)}`)
    }
    throw new Error(`experiment "${experimentName}" resolved to no ID`)
  }

  /**
   * Resolve the experiment ID on an MLflow tracking server over its REST API.
   * @param {string} base - the tracking URI without a trailing slash.
   * @param {string} experimentName - the configured experiment name.
   * @returns {Promise<string>} the experiment ID.
   */
  async function resolveHttpExperiment(base, experimentName) {
    const headers = {}
    if (options.trackingToken !== '') {
      headers.authorization = `Bearer ${options.trackingToken}`
    } else if (process.env.MLFLOW_TRACKING_TOKEN !== undefined && process.env.MLFLOW_TRACKING_TOKEN !== '') {
      headers.authorization = `Bearer ${process.env.MLFLOW_TRACKING_TOKEN}`
    } else if (process.env.MLFLOW_TRACKING_USERNAME !== undefined && process.env.MLFLOW_TRACKING_PASSWORD !== undefined) {
      const pair = `${process.env.MLFLOW_TRACKING_USERNAME}:${process.env.MLFLOW_TRACKING_PASSWORD}`
      headers.authorization = `Basic ${Buffer.from(pair, 'utf8').toString('base64')}`
    }
    const lookup = `${base}/api/2.0/mlflow/experiments/get-by-name?experiment_name=${encodeURIComponent(experimentName)}`
    const found = await fetch(lookup, { headers })
    if (found.ok) {
      const body = await found.json()
      const id = body?.experiment?.experiment_id ?? body?.experiment_id
      if (id !== undefined && id !== null && String(id) !== '') return String(id)
    }
    const created = await fetch(`${base}/api/2.0/mlflow/experiments/create`, {
      method: 'POST',
      headers: { ...headers, 'content-type': 'application/json' },
      body: JSON.stringify({ name: experimentName }),
    })
    if (!created.ok) {
      const detail = await created.text().catch(() => '')
      throw new Error(`cannot create experiment "${experimentName}" at ${base}: HTTP ${created.status} ${clip(detail, 400)}`)
    }
    const body = await created.json()
    const id = body?.experiment_id ?? body?.experiment?.experiment_id
    if (id === undefined || id === null || String(id) === '') {
      throw new Error(`creating experiment "${experimentName}" returned no ID`)
    }
    log.info?.(`${label}: created experiment ${experimentName} (${id})`)
    return String(id)
  }

  /**
   * Resolve where traces go: the tracking target and the experiment ID.
   *
   * A `databricks://profile` value in `trackingUri` names a profile rather than
   * an HTTP endpoint, so a configuration written against an earlier spelling
   * keeps working.
   * @param {boolean} force - refresh the CLI token even when it is still valid.
   * @returns {Promise<{trackingUri: string, experimentId: string, host?: string, token?: string}>} the target.
   */
  async function resolveTarget(force) {
    const configured = options.trackingUri.trim()
    if (configured !== '' && !isDatabricksTracking(configured)) {
      const base = configured.replace(/\/+$/, '')
      const experimentId = options.experimentId !== ''
        ? options.experimentId
        : await resolveHttpExperiment(base, options.experiment)
      return { trackingUri: base, experimentId }
    }
    const profile = configured.startsWith('databricks://')
      ? configured.slice('databricks://'.length)
      : profileName()
    const auth = await resolveCredentials(profile, force)
    credentials = auth
    const experimentId = options.experimentId !== ''
      ? options.experimentId
      : await resolveDatabricksExperiment(options.experiment, auth)
    return { trackingUri: 'databricks', experimentId, host: auth.host, token: auth.token }
  }

  /**
   * Re-initialize the SDK with a fresh token once the current one nears expiry.
   *
   * A re-initialization is visible as a new tracer provider, so it only happens
   * between turns: a trace split across two providers would lose its root. A
   * turn that outlives its token still exports if the token is valid when the
   * turn ends, which the skew budget makes the normal case.
   * @returns {Promise<void>} resolves once the attempt settles.
   */
  async function refreshCredentials() {
    if (!ready || sdk === undefined || credentials === undefined) return
    if (credentials.expiresAt - Date.now() > options.refreshSkewMs) return
    for (const record of sessions.values()) {
      if (record.turn !== undefined) return
    }
    const now = Date.now()
    if (now - lastRefreshAttempt < options.retryIntervalMs) return
    lastRefreshAttempt = now
    try {
      const refreshed = await resolveCredentials(credentials.profile, true)
      sdk.mlflow.init({
        trackingUri: target?.trackingUri ?? 'databricks',
        experimentId: target?.experimentId ?? '',
        host: refreshed.host,
        databricksToken: refreshed.token,
      })
      credentials = refreshed
      log.info?.(`${label}: refreshed the OAuth token for profile "${refreshed.profile}"`)
    } catch (error) {
      log.warn?.(`${label}: token refresh failed: ${describe(error)}`)
    }
  }

  /**
   * Initialize tracing, once per successful target resolution.
   *
   * The first attempt happens during activation so the first turn is traced;
   * later attempts happen at a turn start when the previous attempt failed, at
   * most once per `retryIntervalMs`, so a broken profile costs one lookup per
   * window rather than one per turn.
   * @returns {Promise<boolean>} whether tracing is ready.
   */
  async function ensureReady() {
    if (ready) return true
    const now = Date.now()
    if (lastError !== '' && now - lastAttempt < options.retryIntervalMs) return false
    lastAttempt = now
    try {
      const resolved = await resolveTarget(false)
      const signature = [resolved.trackingUri, resolved.experimentId, resolved.host ?? '', resolved.token ?? ''].join('|')
      const cacheKey = Symbol.for('dsh.databricks-mlflow-tracing.signature')
      if (sdk === undefined) sdk = await loadSdk()
      if (globalThis[cacheKey] !== undefined && globalThis[cacheKey] !== signature) {
        // The target changed, so the SDK has to be initialised again. Release
        // the global tracer provider first: OpenTelemetry refuses to register a
        // second one, and the SDK shuts the previous down without awaiting it,
        // which leaves spans going nowhere and every flush emptying an empty
        // queue. `trace.disable()` is the documented way back to a registrable
        // state.
        try {
          sdk.otel?.trace?.disable?.()
          await new Promise(resolve => setTimeout(resolve, 0))
        } catch {
          // Without the API, the re-initialisation below is no worse than before.
        }
      }
      if (globalThis[cacheKey] === signature) {
        // Already initialised in this process with this exact target: reuse it
        // rather than re-initialising, which would detach the span pipeline.
        target = resolved
        ready = true
        lastError = ''
        return true
      }
      globalThis[cacheKey] = signature
      sdk.mlflow.init({
        trackingUri: resolved.trackingUri,
        experimentId: resolved.experimentId,
        ...(resolved.host !== undefined ? { host: resolved.host, databricksToken: resolved.token } : {}),
        ...(options.trackingToken !== '' ? { trackingServerToken: options.trackingToken } : {}),
      })
      target = resolved
      ready = true
      lastError = ''
      log.info?.(`${label}: tracing turn spans to ${resolved.trackingUri}, experiment ${resolved.experimentId}`)
      return true
    } catch (error) {
      ready = false
      lastError = describe(error)
      log.warn?.(`${label}: tracing is off: ${lastError}`)
      return false
    }
  }

  // --- Span helpers --------------------------------------------------------

  /**
   * Start one span, containing any SDK rejection.
   * @param {object} input - span options (`name`, `spanType`, `parent`, …).
   * @returns {any | undefined} the live span, or undefined when tracing is off.
   */
  function startSpan(input) {
    if (!ready || stopped || sdk === undefined) return undefined
    try {
      return sdk.mlflow.startSpan(input)
    } catch (error) {
      log.warn?.(`${label}: could not start span "${String(input?.name)}": ${describe(error)}`)
      return undefined
    }
  }

  /**
   * End one span, containing any SDK rejection.
   * @param {any | undefined} span - the live span.
   * @param {object} [endOptions] - outputs, attributes, status, and end time.
   * @returns {void}
   */
  function endSpan(span, endOptions) {
    if (span === undefined || span === null) return
    try {
      span.end(endOptions)
    } catch (error) {
      log.warn?.(`${label}: could not end span: ${describe(error)}`)
    }
  }

  /**
   * Run a span mutation, ignoring failures on an ended or unknown span.
   * @param {any | undefined} span - the live span.
   * @param {() => void} change - the mutation.
   * @returns {void}
   */
  function mutateSpan(span, change) {
    if (span === undefined || span === null) return
    try {
      change()
    } catch {
      // Span presentation is best-effort; losing it must not break a turn.
    }
  }

  /**
   * Flush pending exports without awaiting them in the event listener.
   * @returns {void}
   */
  function flush() {
    if (sdk === undefined) return
    try {
      const pending = sdk.mlflow.flushTraces()
      if (pending !== undefined && typeof pending.catch === 'function') {
        pending.catch((error) => log.warn?.(`${label}: trace flush failed: ${describe(error)}`))
      }
    } catch (error) {
      log.warn?.(`${label}: trace flush failed: ${describe(error)}`)
    }
  }

  /**
   * Attach conversation identity to the active trace.
   *
   * MLflow reads `mlflow.trace.session` and `mlflow.trace.user` from trace
   * metadata, which the SDK exposes only through the OpenTelemetry active
   * context. Running the update inside a context that carries the turn's root
   * span is what makes the session grouping appear in the UI.
   * @param {object} record - the per-session trace state.
   * @param {object} turn - the open turn.
   * @param {Record<string, string>} [extraTags] - tags to merge over the defaults.
   * @returns {void}
   */
  function annotateTrace(record, turn, extraTags) {
    if (sdk === undefined || sdk.otel === undefined || turn?.root === undefined) return
    const metadata = {
      'mlflow.trace.session': record.id,
      'mlflow.trace.user': hostUser,
      'mlflow.source.type': 'LOCAL',
      'mlflow.source.name': 'deepseek-harness',
    }
    const tags = {
      'dsh.session.id': record.id,
      'dsh.origin': record.header.origin ?? 'user',
      ...(record.header.agentPreset !== undefined ? { 'dsh.agent.preset': String(record.header.agentPreset) } : {}),
      ...(record.header.cwd !== undefined ? { 'dsh.session.cwd': String(record.header.cwd) } : {}),
      ...(record.header.parentSession !== undefined ? { 'dsh.parent.session': String(record.header.parentSession) } : {}),
      ...(record.header.delegationDepth !== undefined ? { 'dsh.delegation.depth': String(record.header.delegationDepth) } : {}),
      ...(record.title !== undefined ? { 'dsh.session.title': record.title } : {}),
      ...options.tags,
      ...(extraTags ?? {}),
    }
    try {
      const api = sdk.otel
      const active = api.trace.setSpan(api.context.active(), turn.root._span)
      api.context.with(active, () => {
        sdk.mlflow.updateCurrentTrace({ metadata, tags })
      })
    } catch {
      // Session metadata is a convenience; the span tree still exports.
    }
  }

  // --- Session state -------------------------------------------------------

  /**
   * Per-session trace state, created on first sight of the session.
   *
   * Creation metadata comes from `session.header`, which exists for a fresh,
   * resumed, forked, or subagent session alike; the plugin therefore never
   * depends on having observed `session/created`.
   * @param {object} session - the session whose log grew.
   * @returns {object} the trace state for that session.
   */
  function sessionState(session) {
    const id = typeof session?.id === 'string' ? session.id : String(session?.id ?? 'unknown')
    let record = sessions.get(id)
    if (record === undefined) {
      const header = session?.header ?? {}
      record = {
        id,
        header: {
          origin: header.origin === 'subagent' ? 'subagent' : 'user',
          cwd: typeof header.cwd === 'string' ? header.cwd : undefined,
          createdAt: typeof header.createdAt === 'number' ? header.createdAt : undefined,
          agentPreset: typeof header.agentPreset === 'string' ? header.agentPreset : undefined,
          parentSession: header.parentSession,
          delegationDepth: typeof header.delegationDepth === 'number' ? header.delegationDepth : undefined,
        },
        title: undefined,
        turn: undefined,
        lastTurn: 0,
      }
      sessions.set(id, record)
    }
    return record
  }

  /**
   * The span a new child belongs under: the open step, else the turn root.
   * @param {object} turn - the open turn.
   * @returns {any | undefined} the parent span.
   */
  function parentSpan(turn) {
    if (turn.step !== undefined && turn.step.ended !== true) return turn.step.span
    return turn.root
  }

  // --- Turn lifecycle ------------------------------------------------------

  /**
   * Start a turn's root span.
   *
   * A turn whose start arrives while tracing is off is skipped entirely rather
   * than partly recorded: a trace missing its model calls is worse than no
   * trace. A new turn also closes an abandoned predecessor, which is what a
   * crash-recovered log looks like before its synthetic closer.
   * @param {object} record - the per-session trace state.
   * @param {object} event - the `turn/start` event.
   * @returns {void}
   */
  function startTurn(record, event) {
    const data = event.data ?? {}
    const number = Number.isSafeInteger(data.turn) ? data.turn : record.lastTurn + 1
    record.lastTurn = number
    if (record.turn !== undefined) finishTurn(record, 'interrupted', event.time, {})
    if (!ready) {
      record.turn = undefined
      void ensureReady()
      return
    }
    const startTimeNs = toNanoseconds(event.time)
    const root = startSpan({
      name: `${options.traceName} ${number}`,
      spanType: 'AGENT',
      ...(startTimeNs !== undefined ? { startTimeNs } : {}),
      ...(options.captureContent ? { inputs: { messages: [] } } : {}),
      attributes: spanAttributes({
        'dsh.session.id': record.id,
        'dsh.turn': number,
        ...(record.header.cwd !== undefined ? { 'dsh.session.cwd': record.header.cwd } : {}),
        ...(record.header.agentPreset !== undefined ? { 'dsh.agent.preset': record.header.agentPreset } : {}),
        'dsh.origin': record.header.origin,
      }),
    })
    if (root === undefined) {
      record.turn = undefined
      return
    }
    record.turn = {
      number,
      root,
      step: undefined,
      tools: new Map(),
      pendingInputs: [],
      prompts: [],
      promptCount: 0,
      contextKinds: [],
      lastText: '',
      lastTextLength: 0,
      steps: 0,
      llmCalls: 0,
      toolCalls: 0,
      inputTokens: 0,
      outputTokens: 0,
    }
    annotateTrace(record, record.turn)
  }

  /**
   * Start the CHAIN span that covers one model call and its tool executions.
   * @param {object} record - the per-session trace state.
   * @param {object} event - the `step/start` event.
   * @returns {void}
   */
  function startStep(record, event) {
    const turn = record.turn
    if (turn === undefined) return
    const data = event.data ?? {}
    const number = Number.isSafeInteger(data.step) ? data.step : turn.steps + 1
    if (turn.step !== undefined) endStepSpan(turn, event.time, 'OK')
    const startTimeNs = toNanoseconds(event.time)
    const span = startSpan({
      name: `step ${number}`,
      spanType: 'CHAIN',
      parent: turn.root,
      ...(startTimeNs !== undefined ? { startTimeNs } : {}),
    })
    turn.steps += 1
    turn.step = {
      number,
      span,
      startNs: startTimeNs,
      messages: turn.pendingInputs.splice(0),
      ended: span === undefined,
    }
  }

  /**
   * Close the open step span.
   * @param {object} turn - the open turn.
   * @param {number | undefined} at - the closing event time, in milliseconds.
   * @param {string} status - the span status to record.
   * @returns {void}
   */
  function endStepSpan(turn, at, status) {
    const step = turn.step
    if (step === undefined) return
    turn.step = undefined
    if (step.ended === true) return
    step.ended = true
    const endTimeNs = toNanoseconds(at)
    endSpan(step.span, {
      ...(endTimeNs !== undefined ? { endTimeNs } : {}),
      status: spanStatus(status),
      attributes: spanAttributes({ 'dsh.step': step.number }),
    })
  }

  /**
   * Record one human or injected message entering the open step.
   *
   * Injected context (tool catalogs, file notices, skill lists) is a large
   * fraction of a request but is not what the conversation is about, so only a
   * human prompt becomes an LLM-span input message; the kinds of context that
   * accompanied it are recorded as an attribute instead.
   * @param {object} record - the per-session trace state.
   * @param {object} event - the `user/message` event.
   * @returns {void}
   */
  function addUserMessage(record, event) {
    const turn = record.turn
    if (turn === undefined) return
    const data = event.data ?? {}
    const sourceKind = typeof data.source?.kind === 'string' ? data.source.kind : 'unknown'
    const text = readContent(data.content, false).text
    if (text === '') return
    if (sourceKind === 'user') {
      turn.promptCount += 1
      if (!options.captureContent) return
      const prompt = clip(text, options.maxChars)
      turn.prompts.push(prompt)
      if (turn.step !== undefined) turn.step.messages.push({ role: 'user', content: prompt })
    } else if (!turn.contextKinds.includes(sourceKind)) {
      turn.contextKinds.push(sourceKind)
    }
  }

  /**
   * Record the model call that closed an open step.
   * @param {object} record - the per-session trace state.
   * @param {object} event - the `assistant/message` event.
   * @returns {void}
   */
  function addAssistantMessage(record, event) {
    const turn = record.turn
    if (turn === undefined) return
    const data = event.data ?? {}
    const step = turn.step
    const provider = typeof data.message?.source?.provider === 'string' ? data.message.source.provider : undefined
    const model = typeof data.message?.source?.model === 'string' ? data.message.source.model : undefined
    const parts = readContent(data.message?.content, options.includeReasoning)
    const startedNs = streamStartNs(data.stream, step?.startNs ?? toNanoseconds(event.time))
    const endedNs = toNanoseconds(event.time)
    const inputs = options.captureContent
      ? { messages: (step?.messages ?? []).map(message => ({ role: message.role, content: message.content })) }
      : undefined
    const usage = data.usage ?? {}
    const inputTokens = Number(usage.inputTokens ?? 0) + Number(usage.cacheReadTokens ?? 0) + Number(usage.cacheWriteTokens ?? 0)
    const outputTokens = Number(usage.outputTokens ?? 0)
    const totalTokens = Number.isFinite(Number(usage.totalTokens)) ? Number(usage.totalTokens) : inputTokens + outputTokens
    const attributes = spanAttributes({
      ...(provider !== undefined ? { 'gen_ai.system': provider, 'dsh.llm.provider': provider } : {}),
      ...(model !== undefined ? { 'gen_ai.request.model': model, 'dsh.llm.model': model } : {}),
      'mlflow.chat.tokenUsage': {
        input_tokens: inputTokens,
        output_tokens: outputTokens,
        total_tokens: totalTokens,
      },
      'dsh.tokens.input': inputTokens,
      'dsh.tokens.output': outputTokens,
      'dsh.tokens.total': totalTokens,
      ...(usage.cacheReadTokens !== undefined ? { 'dsh.tokens.cache_read': Number(usage.cacheReadTokens) } : {}),
      ...(usage.cacheWriteTokens !== undefined ? { 'dsh.tokens.cache_write': Number(usage.cacheWriteTokens) } : {}),
      ...(usage.reasoningTokens !== undefined ? { 'dsh.tokens.reasoning': Number(usage.reasoningTokens) } : {}),
      ...(data.interrupted === true ? { 'dsh.interrupted': true } : {}),
      'dsh.step': step?.number ?? 0,
      ...(turn.contextKinds.length > 0 ? { 'dsh.context.kinds': turn.contextKinds.join(',') } : {}),
    })
    const outputs = options.captureContent
      ? {
          text: clip(parts.text, options.maxChars),
          ...(parts.reasoning !== '' ? { reasoning: clip(parts.reasoning, options.maxChars) } : {}),
          ...(parts.toolCalls.length > 0
            ? {
                tool_calls: parts.toolCalls.map(call => ({
                  name: call.name,
                  arguments: parseArguments(call.arguments, options.maxToolResultChars),
                })),
              }
            : {}),
        }
      : undefined
    const span = startSpan({
      name: model !== undefined ? `llm ${model}` : 'llm',
      spanType: 'LLM',
      parent: parentSpan(turn),
      ...(startedNs !== undefined ? { startTimeNs: startedNs } : {}),
      ...(inputs !== undefined ? { inputs } : {}),
      attributes,
    })
    turn.llmCalls += 1
    turn.inputTokens += inputTokens
    turn.outputTokens += outputTokens
    turn.lastTextLength = parts.text.length
    if (options.captureContent && parts.text !== '') turn.lastText = clip(parts.text, options.maxChars)
    endSpan(span, {
      ...(outputs !== undefined ? { outputs } : {}),
      ...(endedNs !== undefined ? { endTimeNs: endedNs } : {}),
      status: spanStatus(data.interrupted === true ? 'UNSET' : 'OK'),
    })
  }

  /**
   * Record a model attempt that committed no surface message.
   *
   * These are exactly the failures, retries, and cancellations a user asks
   * about later, so they get their own ERROR span rather than disappearing with
   * the retry that replaced them.
   * @param {object} record - the per-session trace state.
   * @param {object} event - the `assistant/attempt` event.
   * @returns {void}
   */
  function addAttempt(record, event) {
    const turn = record.turn
    if (turn === undefined) return
    const data = event.data ?? {}
    const step = turn.step
    const startedNs = streamStartNs(data.stream, step?.startNs ?? toNanoseconds(event.time))
    const span = startSpan({
      name: 'llm attempt',
      spanType: 'LLM',
      parent: parentSpan(turn),
      ...(startedNs !== undefined ? { startTimeNs: startedNs } : {}),
      attributes: spanAttributes({
        'dsh.attempt.records': Array.isArray(data.stream) ? data.stream.length : 0,
        'dsh.step': step?.number ?? 0,
      }),
    })
    turn.llmCalls += 1
    endSpan(span, {
      outputs: { attempt: 'failed' },
      ...(toNanoseconds(event.time) !== undefined ? { endTimeNs: toNanoseconds(event.time) } : {}),
      status: spanStatus('ERROR'),
    })
  }

  /**
   * Start a TOOL span for one requested invocation.
   * @param {object} record - the per-session trace state.
   * @param {object} event - the `tool/call` event.
   * @returns {void}
   */
  function startTool(record, event) {
    const turn = record.turn
    if (turn === undefined) return
    const data = event.data ?? {}
    const callId = typeof data.callId === 'string' ? data.callId : String(data.callId ?? '')
    if (callId === '') return
    const startedNs = toNanoseconds(event.time)
    const span = startSpan({
      name: `tool ${typeof data.name === 'string' ? data.name : 'call'}`,
      spanType: 'TOOL',
      parent: parentSpan(turn),
      ...(startedNs !== undefined ? { startTimeNs: startedNs } : {}),
      ...(options.captureContent
        ? { inputs: { name: data.name, arguments: parseArguments(data.arguments, options.maxToolResultChars) } }
        : {}),
      attributes: spanAttributes({ 'dsh.tool.name': String(data.name ?? ''), 'dsh.step': turn.step?.number ?? 0 }),
    })
    turn.toolCalls += 1
    turn.tools.set(callId, { span, name: String(data.name ?? 'call'), ended: false })
  }

  /**
   * Close the TOOL span a result answers.
   * @param {object} record - the per-session trace state.
   * @param {object} event - the `tool/result` event.
   * @returns {void}
   */
  function finishTool(record, event) {
    const turn = record.turn
    if (turn === undefined) return
    const data = event.data ?? {}
    const callId = typeof data.message?.toolCallId === 'string'
      ? data.message.toolCallId
      : typeof data.message?.source?.callId === 'string'
        ? data.message.source.callId
        : ''
    const tool = callId !== '' ? turn.tools.get(callId) : undefined
    if (tool === undefined) return
    turn.tools.delete(callId)
    tool.ended = true
    const isError = data.message?.isError === true || data.error !== undefined
    const text = options.captureContent ? clip(readContent(data.message?.content, false).text, options.maxToolResultChars) : ''
    const attributes = spanAttributes({
      ...(data.error?.code !== undefined ? { 'dsh.tool.error_code': String(data.error.code) } : {}),
      ...(data.error?.reason !== undefined ? { 'dsh.tool.error_reason': clip(String(data.error.reason), 400) } : {}),
    })
    endSpan(tool.span, {
      ...(options.captureContent ? { outputs: { content: text, is_error: isError } } : { outputs: { is_error: isError } }),
      ...(toNanoseconds(event.time) !== undefined ? { endTimeNs: toNanoseconds(event.time) } : {}),
      status: spanStatus(isError ? 'ERROR' : 'OK'),
      ...(Object.keys(attributes).length > 0 ? { attributes } : {}),
    })
    if (options.captureContent && text !== '') {
      turn.pendingInputs.push({ role: 'tool', tool_call_id: callId, content: text })
    }
  }

  /**
   * Close one turn, ending anything still open inside it.
   *
   * The order matters: children close before the root, because the MLflow SDK
   * exports a trace when its root span ends.
   * @param {object} record - the per-session trace state.
   * @param {string} kind - the turn end reason kind.
   * @param {number | undefined} at - the closing event time, in milliseconds.
   * @param {object} detail - extra facts about the reason (`cause`, `error`).
   * @returns {void}
   */
  function finishTurn(record, kind, at, detail) {
    const turn = record.turn
    if (turn === undefined) return
    record.turn = undefined
    const endTimeNs = toNanoseconds(at)
    for (const [callId, tool] of turn.tools) {
      if (tool.ended === true) continue
      endSpan(tool.span, {
        outputs: { interrupted: true },
        status: spanStatus('ERROR'),
        ...(endTimeNs !== undefined ? { endTimeNs } : {}),
      })
      turn.tools.delete(callId)
    }
    endStepSpan(turn, at, kind === 'error' ? 'ERROR' : 'OK')

    if (options.captureContent) {
      mutateSpan(turn.root, () => {
        turn.root.setInputs({
          messages: turn.prompts.map(text => ({ role: 'user', content: text })),
        })
        turn.root.setOutputs({
          text: turn.lastText,
          ...(turn.promptCount === 0 ? { note: 'no human prompt in this turn' } : {}),
        })
      })
    } else {
      mutateSpan(turn.root, () => {
        turn.root.setInputs({ prompts: turn.promptCount })
        turn.root.setOutputs({ text_length: turn.lastTextLength })
      })
    }

    const abortCause = typeof detail?.cause?.kind === 'string' ? detail.cause.kind : undefined
    const failure = detail?.error
    mutateSpan(turn.root, () => {
      turn.root.setAttributes(spanAttributes({
        'dsh.turn.reason': kind,
        'dsh.turn.steps': turn.steps,
        'dsh.turn.llm_calls': turn.llmCalls,
        'dsh.turn.tool_calls': turn.toolCalls,
        'dsh.turn.tokens.input': turn.inputTokens,
        'dsh.turn.tokens.output': turn.outputTokens,
        ...(abortCause !== undefined ? { 'dsh.turn.abort_cause': abortCause } : {}),
        ...(failure !== undefined ? { 'dsh.turn.error': clip(describe(failure), 1000) } : {}),
      }))
    })

    const status = kind === 'error' ? 'ERROR' : 'OK'
    annotateTrace(record, turn, { 'dsh.turn.reason': kind })
    const traceId = typeof turn.root.traceId === 'string' ? turn.root.traceId : undefined
    endSpan(turn.root, {
      ...(endTimeNs !== undefined ? { endTimeNs } : {}),
      status: spanStatus(status),
    })
    if (traceId !== undefined) {
      log.info?.(`${label}: turn ${turn.number} of ${record.id} -> ${traceId} (${turn.steps} steps, ${turn.llmCalls} model calls, ${turn.toolCalls} tool calls)`)
    }
    flush()
    void refreshCredentials()
  }

  // --- Event projection ----------------------------------------------------

  /**
   * Project one committed session event into the trace.
   * @param {object} session - the session whose log grew.
   * @param {object} event - the appended event, exactly as recorded.
   * @returns {void}
   */
  function observe(session, event) {
    if (stopped || event === null || typeof event !== 'object') return
    if (event.type === 'session/title') {
      const record = sessionState(session)
      const title = event.data?.title
      if (typeof title === 'string' && title !== '') {
        record.title = title
        if (record.turn !== undefined) annotateTrace(record, record.turn, { 'dsh.session.title': title })
      }
      return
    }
    const record = sessionState(session)
    switch (event.type) {
      case 'turn/start':
        startTurn(record, event)
        return
      case 'step/start':
        startStep(record, event)
        return
      case 'user/message':
        addUserMessage(record, event)
        return
      case 'assistant/message':
        addAssistantMessage(record, event)
        return
      case 'assistant/attempt':
        addAttempt(record, event)
        return
      case 'tool/call':
        startTool(record, event)
        return
      case 'tool/result':
        finishTool(record, event)
        return
      case 'step/end':
        if (record.turn !== undefined) endStepSpan(record.turn, event.time, 'OK')
        return
      case 'turn/end': {
        const reason = event.data?.reason ?? {}
        finishTurn(record, typeof reason.kind === 'string' ? reason.kind : 'completed', event.time, {
          cause: reason.reason,
          error: reason.error,
        })
        return
      }
      default:
        return
    }
  }

  ctx.on('session/event', (session, event) => {
    try {
      observe(session, event)
    } catch (error) {
      log.warn?.(`${label}: skipped one session event: ${describe(error)}`)
    }
  })

  ctx.on('session/disposed', (session) => {
    const id = typeof session?.id === 'string' ? session.id : String(session?.id ?? '')
    const record = sessions.get(id)
    if (record === undefined) return
    if (record.turn !== undefined) finishTurn(record, 'interrupted', Date.now(), { cause: { kind: 'disposed' } })
    sessions.delete(id)
  })

  ctx.on('dispose', () => {
    for (const record of sessions.values()) {
      if (record.turn !== undefined) finishTurn(record, 'interrupted', Date.now(), { cause: { kind: 'disposed' } })
    }
    sessions.clear()
    stopped = true
    flush()
  })

  await ensureReady()
  if (!ready) {
    log.warn?.(`${label}: set profile/experiment in the plugin configuration; tracing retries at the next turn start`)
  }
}
