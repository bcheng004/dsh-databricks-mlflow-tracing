# @local/dsh-databricks-mlflow-tracing

Sends every DeepSeek Harness conversation turn to MLflow as a trace, through a
Databricks profile.

```
dsh turn 2            AGENT   the turn: prompt in, final answer out
├── step 1            CHAIN
│   └── llm <model>   LLM     inputs, outputs, token usage
├── step 2            CHAIN
│   ├── llm <model>   LLM
│   └── tool bash     TOOL    arguments and result
└── step 3            CHAIN
    └── llm <model>   LLM
```

Traces land in the MLflow experiment of the configured Databricks workspace and
are grouped by conversation through MLflow's `mlflow.trace.session` metadata, so
one session reads as a sequence of turns in the MLflow UI.

## Why this exists

The Harness already writes a complete, ordered session log: every turn boundary,
step, model call, tool call, and result is committed before anything else sees
it. That log is exactly the shape of a trace, and MLflow is a natural place to
read it back — including for sessions that ran on another machine and were
resumed here, and for subagent sessions that never had a window of their own.

This plugin is a projection, not an instrumentation library: it does not hook the
model adapter or the tool pipeline, so a trace can never contain something the
session did not commit, and a cancelled turn still closes its spans.

## Install

### Before you start

- **DeepSeek Harness 0.2.2 or newer.** The bundle declares
  `engines.dsh: ">=0.2.2-rc.3 <0.3.0"`, so an older build, or a 0.3 build, is
  refused while installing instead of failing at request time.
- **A signed-in Databricks CLI.** Run `databricks auth login --profile
  <profile>` once. The plugin asks the CLI for an access token for the named
  profile, so a machine where two profiles point at the same workspace still
  works — the CLI is never asked to guess by host.
- **An MLflow experiment.** The default, `/Users/<you>/dsh-conversations`, is
  created on first use if it does not exist. Databricks workspace permission to
  create an experiment in your home folder is enough.

### 1. Add the bundle

**Plugins → Add plugin** accepts a package name, Git address, tarball, or
absolute local path:

```
github:bcheng004/dsh-databricks-mlflow-tracing
```

A package tarball works too, and is the shape this bundle needs: the plugin has
runtime dependencies (`mlflow-tracing`, `@databricks/sdk-experimental`,
`@opentelemetry/api`), and pnpm installs them from the tarball's manifest. A
plain directory added through **Add plugin** is linked as-is, so its
`node_modules` must already exist — run `npm install` in the checkout first. A
code change in a linked checkout is picked up at the next profile recomposition
(saving the profile patch, or a Plugins page operation), not on the next turn.

### 2. Configure it for your workspace

The bundle ships no machine-specific values, so name yours in your own profile's
`cordis.patch.yml`. An override replaces the row's complete configuration, so
list every field you need:

```yaml
- id: databricks-mlflow-tracing
  name: '@local/dsh-databricks-mlflow-tracing'
  config:
    profile: my-workspace                 # ~/.databrickscfg profile to trace into
    experiment: dsh-conversations         # relative → /Users/<you>/dsh-conversations
```

`profile` is the one that almost always differs: an empty value follows
`DATABRICKS_CONFIG_PROFILE`, else the `DEFAULT` entry in `~/.databrickscfg`, and
a wrong value silently sends traces to another workspace. A relative
`experiment` name is resolved against the current user; an absolute workspace
path such as `/Shared/agent-traces/dsh` is used as written.

`cliPath` matters on a GUI-launched Desktop app, whose `PATH` is the launcher's
minimal one rather than your login shell's. The plugin tries `databricks` on
`PATH` and then `/opt/homebrew/bin`, `/usr/local/bin`, and `~/.local/bin`, which
covers a normal macOS install; set the absolute path when the CLI lives
elsewhere (Intel Macs usually use `/usr/local/bin/databricks`).

### 3. Verify a trace arrived

Send one message, then read the application log for the line the plugin writes at
the end of each turn:

```
databricks-mlflow-tracing[my-workspace]: turn 1 of session-… -> tr-… (3 steps, 3 model calls, 4 tool calls)
```

The trace ID resolves in the experiment:

```sh
databricks auth token -p my-workspace -o json | jq -r .access_token > /tmp/databricks-token
curl -s -H "Authorization: Bearer $(cat /tmp/databricks-token)" \
  "https://<workspace-host>/api/3.0/mlflow/traces/<trace-id>" | jq '.trace.trace_info.state'
```

An empty experiment after a turn means the export failed; the application log
carries the SDK's own message next to the plugin's warning. A turn with no trace
line never started tracing: initialization failed, and the log says why — a
missing or expired profile usually reports the CLI's own
`databricks auth login --profile …` advice.

## What a trace records

| Span | Type | Inputs | Outputs |
|---|---|---|---|
| `dsh turn <n>` | AGENT | the human prompts of the turn | the last assistant text |
| `step <n>` | CHAIN | — | — |
| `llm <model>` | LLM | the prompt and the tool results that entered the step | assistant text, reasoning, requested tool calls |
| `llm attempt` | LLM | — | `{ attempt: "failed" }`, status ERROR |
| `tool <name>` | TOOL | tool name and parsed arguments | result text, `is_error` |

Every LLM span carries `mlflow.chat.tokenUsage` (input, output, total) plus the
cache and reasoning token counts the adapter reported, and `gen_ai.system` /
`gen_ai.request.model`. The root span carries the turn's step, model-call, tool-call,
and token totals. Trace tags record `dsh.session.id`, `dsh.session.title` (when
the harness titles the session), `dsh.session.cwd`, `dsh.agent.preset`,
`dsh.origin` (`user` or `subagent`), the parent session and delegation depth for
subagents, and `dsh.turn.reason` (`completed`, `aborted`, `error`, `interrupted`).
A turn that ended in an error is exported as an error trace.

Injected context — the runtime reminder, skill catalogs, time notes — is not
recorded as a prompt; the kinds that accompanied a step are listed in the LLM
span's `dsh.context.kinds` attribute instead, because that text is a large
fraction of a request but not what the conversation is about.

## Configuration

| Field | Default | Meaning |
|---|---|---|
| `profile` | `''` | Databricks profile that supplies the workspace and the OAuth token; empty follows `DATABRICKS_CONFIG_PROFILE`, else `DEFAULT` |
| `experiment` | `dsh-conversations` | Experiment name; a relative name lands under `/Users/<you>/` |
| `experimentId` | `''` | Experiment ID; when set, no experiment lookup happens at all |
| `trackingUri` | `''` | An OSS MLflow server URL; empty means the Databricks workspace of `profile`. `databricks://<profile>` is accepted as a profile spelling |
| `databricksConfigPath` | `''` | Alternate `~/.databrickscfg` location |
| `cliPath` | `''` | Absolute path to the Databricks CLI; empty searches `PATH` and the usual install locations |
| `trackingToken` | `''` | Bearer token for an OSS MLflow server; empty reads `MLFLOW_TRACKING_TOKEN` |
| `captureContent` | `true` | Record prompts, model output, reasoning, arguments, and tool results |
| `includeReasoning` | `true` | Keep assistant reasoning in the LLM span |
| `maxChars` | `20000` | Character budget for one prompt, response, or reasoning field |
| `maxToolResultChars` | `4000` | Character budget for one tool result or argument payload |
| `traceName` | `dsh turn` | Root span name prefix; the turn number is appended |
| `refreshSkewMs` | `600000` | Replace the OAuth access token once it is this close to expiry |
| `timeoutMs` | `20000` | Kill one Databricks CLI call after this long |
| `retryIntervalMs` | `300000` | How long a failed initialization or token refresh waits before the next attempt |
| `tags` | `{}` | Extra trace tags applied to every trace |

### Tracing an OSS MLflow server

Set `trackingUri` to the server URL and leave `profile` empty; no Databricks CLI
or profile is needed:

```yaml
- id: databricks-mlflow-tracing
  name: '@local/dsh-databricks-mlflow-tracing'
  config:
    trackingUri: http://localhost:5000
    experiment: dsh-conversations
    trackingToken: ''            # or MLFLOW_TRACKING_TOKEN in the environment
```

## What leaves the machine

- The **tracking target and experiment** only. The plugin sends trace data to the
  MLflow experiment you configure; it makes no model requests and reads nothing
  outside the session log, `~/.databrickscfg`, and the Databricks CLI.
- **Session content**, when `captureContent` is on (the default): prompts,
  assistant text, reasoning, tool arguments, and tool results. Turn it off and
  only structure, timing, model identity, and token counts are recorded.
- Every captured text field is truncated to its configured budget, and a
  truncated value says so in place. There is no separate redaction policy: if a
  secret appears in a prompt or a tool result, it appears in the trace, so treat
  the experiment with the same access controls as the workspace itself.

## Failure behavior

- **A failed sign-in or a missing experiment** is logged once and retried at the
  next turn start, at most once per `retryIntervalMs`. The conversation is never
  affected; a turn that could not be traced is skipped entirely rather than
  recorded without its model calls.
- **An expiring token** is replaced between turns, never inside one, so a trace
  cannot be split across two SDK generations. If the refresh fails, the current
  token keeps working until it expires and the refresh is retried at a later
  turn, at most once per `retryIntervalMs`.
- **A failed export** is reported by the MLflow SDK in the application log. The
  turn's spans are dropped with it; nothing is queued for a later run.
- **Turning the plugin off** stops tracing at once. Turns already exported stay
  in MLflow.

## How it works

`session/event` is a post-commit feed: every appended event is delivered to
listeners after the log push, with the event exactly as recorded, and a listener
failure cannot fail the append. The plugin folds that feed per session:
`turn/start` opens the root span, `step/start` a CHAIN span, `assistant/message`
an LLM span whose start comes from the embedded stream's first frame,
`tool/call` and `tool/result` a TOOL span, and `turn/end` closes everything and
flushes the trace.

Parenting is explicit (`startSpan({ parent })`) rather than ambient OpenTelemetry
context, because a turn's spans are opened and closed by unrelated callbacks. The
MLflow SDK attaches no trace metadata without an active span, so the plugin
updates `mlflow.trace.session` and its tags inside a context that carries the
turn's root span.

Session identity comes from `session.header`, which exists for fresh, resumed,
forked, and subagent sessions alike, so nothing depends on having observed a
session's creation. Only events that occur while the plugin runs are traced: a
resumed session's earlier turns are already in the log and are not re-exported.

Credentials are read per profile, explicitly: the host comes from
`~/.databrickscfg` (or `DATABRICKS_HOST`), and the access token comes from
`databricks auth token -p <profile>`. Both are handed to the MLflow SDK as
explicit values rather than through its profile lookup, because that lookup
shells out to `databricks auth token --host <host>` and the CLI refuses when two
profiles name the same workspace. The token is replaced about ten minutes before
it expires, between turns.

### Notes for a shared deployment

- **Every user needs their own profile.** OAuth user-to-machine profiles are
  per-user and per-machine; the plugin does not ship credentials and cannot mint
  them for someone else.
- **`@local/` is a placeholder package name**, not a published scope. It works
  from a Git or tarball install; publishing to npm would mean renaming the
  package and the row reference together.
- **Traces accumulate in one experiment.** Use `experiment` to separate
  deployments, or `tags` to separate environments, and prune the experiment the
  same way you would any other MLflow data.
