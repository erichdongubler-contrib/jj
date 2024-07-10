// Copyright 2024 The Jujutsu Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use std::collections::{HashMap, HashSet};
use std::io::Write;
use std::process::Stdio;
use std::sync::mpsc::channel;

use config::ConfigError;
use futures::StreamExt;
use itertools::Itertools;
use jj_lib::backend::{BackendError, BackendResult, CommitId, FileId, TreeValue};
use jj_lib::matchers::{EverythingMatcher, Matcher};
use jj_lib::merged_tree::MergedTreeBuilder;
use jj_lib::repo::Repo;
use jj_lib::repo_path::RepoPathBuf;
use jj_lib::revset::{RevsetExpression, RevsetIteratorExt};
use jj_lib::store::Store;
use pollster::FutureExt;
use rayon::iter::IntoParallelIterator;
use rayon::prelude::ParallelIterator;
use tracing::instrument;

use crate::cli_util::{CommandHelper, RevisionArg, WorkspaceCommandHelper};
use crate::command_error::CommandError;
use crate::config::{CommandNameAndArgs, NonEmptyCommandArgsVec};
use crate::ui::Ui;

/// Update files with formatting fixes or other changes
///
/// The primary use case for this command is to apply the results of automatic
/// code formatting tools to revisions that may not be properly formatted yet.
/// It can also be used to modify files with other tools like `sed` or `sort`.
///
/// The changed files in the given revisions will be updated with any fixes
/// determined by passing their file content through any external tools the user
/// has configured for those files. Descendants will also be updated by passing
/// their versions of the same files through external tools, which will ensure
/// that the fixes are not lost. This will never result in new conflicts. Files
/// with existing conflicts will be updated on all sides of the conflict, which
/// can potentially increase or decrease the number of conflict markers.
///
/// The external tools must accept the current file content on standard input,
/// and return the updated file content on standard output. A tool's output will
/// not be used unless it exits with a successful exit code. Output on standard
/// error will be passed through to the terminal.
///
/// Tools are defined in a table, and are applied in the same order as they are
/// defined. If two tools affect the same file, the second tool will receive its
/// input from the output of the first tool. The attributes of each tool are:
///  - `name`: An arbitrary unique identifier for the tool, which will be used
///    to identify it in error messages. If two or more tools have the same name
///    configured, `jj fix` will fail without making any changes.
///  - `command`: The arguments used to run the tool. The first argument is the
///    path to an executable file. Arguments can contain the substring `$path`,
///    which will be replaced with the repo-relative path of the file being
///    fixed. It is useful to provide the path to tools that include the path in
///    error messages, or behave differently based on the directory or file
///    name.
///  - `patterns`: Determines which files the tool will affect. If this list is
///    empty, no files will be affected by the tool. If there are multiple
///    patterns, the tool is applied only once to each file in the union of the
///    patterns.
///
/// For example, the following configuration defines how two code formatters
/// (`clang-format` and `black`) will apply to three different file extensions
/// (.cc, .h, and .py):
///
/// [[fix.tools]]
/// name = "clang-format"
/// command = ["clang-format", "--assume-filename=$path"]
/// patterns = ["glob:\"**/*.cc\"",
///             "glob:\"**/*.h\""]
///
/// [[fix.tools]]
/// name = "black"
/// command = ["black", "-", "--stdin-filename=$path"]
/// patterns = ["glob:\"**/*.py"]
///
/// There is also a deprecated configuration schema that defines a single
/// command that will affect all changed files in the specified revisions. For
/// example, the following configuration would apply the Rust formatter to all
/// changed files (whether they are Rust files or not):
///
/// [fix]
/// tool-command = ["rustfmt", "--emit", "stdout"]
///
/// The tool defined by `tool-command` acts as if it was the first entry in
/// `fix.tools`, and uses `pattern = "all()"``. Support for `tool-command`
/// will be removed in a future version.
#[derive(clap::Args, Clone, Debug)]
#[command(verbatim_doc_comment)]
pub(crate) struct FixArgs {
    /// Fix files in the specified revision(s) and their descendants. If no
    /// revisions are specified, this defaults to the `revsets.fix` setting, or
    /// `reachable(@, mutable())` if it is not set.
    #[arg(long, short)]
    source: Vec<RevisionArg>,
    /// Fix only these paths
    #[arg(value_hint = clap::ValueHint::AnyPath)]
    paths: Vec<String>,
}

#[instrument(skip_all)]
pub(crate) fn cmd_fix(
    ui: &mut Ui,
    command: &CommandHelper,
    args: &FixArgs,
) -> Result<(), CommandError> {
    let mut workspace_command = command.workspace_helper(ui)?;
    let tools_config = get_tools_config(&workspace_command, command.settings().config())?;
    let root_commits: Vec<CommitId> = if args.source.is_empty() {
        workspace_command.parse_revset(&RevisionArg::from(
            command.settings().config().get_string("revsets.fix")?,
        ))?
    } else {
        workspace_command.parse_union_revsets(&args.source)?
    }
    .evaluate_to_commit_ids()?
    .collect();
    workspace_command.check_rewritable(root_commits.iter())?;
    let matcher = workspace_command
        .parse_file_patterns(&args.paths)?
        .to_matcher();

    let mut tx = workspace_command.start_transaction();

    // Collect all of the unique `ToolInput`s we're going to use. Tools should be
    // deterministic, and should not consider outside information, so it is safe to
    // deduplicate inputs that correspond to multiple files or commits. This is
    // typically more efficient, but it does prevent certain use cases like
    // providing commit IDs as inputs to be inserted into files. We also need to
    // record the mapping between tool inputs and paths/commits, to efficiently
    // rewrite the commits later.
    //
    // If a path is being fixed in a particular commit, it must also be fixed in all
    // that commit's descendants. We do this as a way of propagating changes,
    // under the assumption that it is more useful than performing a rebase and
    // risking merge conflicts. In the case of code formatters, rebasing wouldn't
    // reliably produce well formatted code anyway. Deduplicating inputs helps
    // to prevent quadratic growth in the number of tool executions required for
    // doing this in long chains of commits with disjoint sets of modified files.
    let commits: Vec<_> = RevsetExpression::commits(root_commits.to_vec())
        .descendants()
        .evaluate_programmatic(tx.base_repo().as_ref())?
        .iter()
        .commits(tx.repo().store())
        .try_collect()?;
    let mut unique_tool_inputs: HashSet<ToolInput> = HashSet::new();
    let mut commit_paths: HashMap<CommitId, HashSet<RepoPathBuf>> = HashMap::new();
    for commit in commits.iter().rev() {
        let mut paths: HashSet<RepoPathBuf> = HashSet::new();

        // Fix all paths that were fixed in ancestors, so we don't lose those changes.
        // We do this instead of rebasing onto those changes, to avoid merge conflicts.
        for parent_id in commit.parent_ids() {
            if let Some(parent_paths) = commit_paths.get(parent_id) {
                paths.extend(parent_paths.iter().cloned());
            }
        }

        // Also fix any new paths that were changed in this commit.
        let tree = commit.tree()?;
        let parent_tree = commit.parent_tree(tx.repo())?;
        let mut diff_stream = parent_tree.diff_stream(&tree, &matcher);
        async {
            while let Some((repo_path, diff)) = diff_stream.next().await {
                let (_before, after) = diff?;
                // Deleted files have no file content to fix, and they have no terms in `after`,
                // so we don't add any tool inputs for them. Conflicted files produce one tool
                // input for each side of the conflict.
                for term in after.into_iter().flatten() {
                    // We currently only support fixing the content of normal files, so we skip
                    // directories and symlinks, and we ignore the executable bit.
                    if let TreeValue::File { id, executable: _ } = term {
                        // TODO: Consider filename arguments and tool configuration instead of
                        // passing every changed file into the tool. Otherwise, the tool has to
                        // be modified to implement that kind of stuff.
                        let tool_input = ToolInput {
                            file_id: id.clone(),
                            repo_path: repo_path.clone(),
                        };
                        unique_tool_inputs.insert(tool_input.clone());
                        paths.insert(repo_path.clone());
                    }
                }
            }
            Ok::<(), BackendError>(())
        }
        .block_on()?;

        commit_paths.insert(commit.id().clone(), paths);
    }

    // Run the configured tool on all of the chosen inputs.
    let fixed_file_ids = fix_file_ids(
        tx.repo().store().as_ref(),
        &tools_config,
        &unique_tool_inputs,
    )?;

    // Substitute the fixed file IDs into all of the affected commits. Currently,
    // fixes cannot delete or rename files, change the executable bit, or modify
    // other parts of the commit like the description.
    let mut num_checked_commits = 0;
    let mut num_fixed_commits = 0;
    tx.mut_repo().transform_descendants(
        command.settings(),
        root_commits.iter().cloned().collect_vec(),
        |mut rewriter| {
            // TODO: Build the trees in parallel before `transform_descendants()` and only
            // keep the tree IDs in memory, so we can pass them to the rewriter.
            let repo_paths = commit_paths.get(rewriter.old_commit().id()).unwrap();
            let old_tree = rewriter.old_commit().tree()?;
            let mut tree_builder = MergedTreeBuilder::new(old_tree.id().clone());
            let mut changes = 0;
            for repo_path in repo_paths {
                let old_value = old_tree.path_value(repo_path)?;
                let new_value = old_value.map(|old_term| {
                    if let Some(TreeValue::File { id, executable }) = old_term {
                        let tool_input = ToolInput {
                            file_id: id.clone(),
                            repo_path: repo_path.clone(),
                        };
                        if let Some(new_id) = fixed_file_ids.get(&tool_input) {
                            return Some(TreeValue::File {
                                id: new_id.clone(),
                                executable: *executable,
                            });
                        }
                    }
                    old_term.clone()
                });
                if new_value != old_value {
                    tree_builder.set_or_remove(repo_path.clone(), new_value);
                    changes += 1;
                }
            }
            num_checked_commits += 1;
            if changes > 0 {
                num_fixed_commits += 1;
                let new_tree = tree_builder.write_tree(rewriter.mut_repo().store())?;
                let builder = rewriter.reparent(command.settings())?;
                builder.set_tree_id(new_tree).write()?;
            }
            Ok(())
        },
    )?;
    writeln!(
        ui.status(),
        "Fixed {num_fixed_commits} commits of {num_checked_commits} checked."
    )?;
    tx.finish(ui, format!("fixed {num_fixed_commits} commits"))
}

/// Represents the API between `jj fix` and the tools it runs.
// TODO: Add the set of changed line/byte ranges, so those can be passed into code formatters via
// flags. This will help avoid introducing unrelated changes when working on code with out of date
// formatting.
#[derive(PartialEq, Eq, Hash, Clone)]
struct ToolInput {
    /// File content is the primary input, provided on the tool's standard
    /// input. We use the `FileId` as a placeholder here, so we can hold all
    /// the inputs in memory without also holding all the content at once.
    file_id: FileId,

    /// The path is provided to allow passing it into the tool so it can
    /// potentially:
    ///  - Choose different behaviors for different file names, extensions, etc.
    ///  - Update parts of the file's content that should be derived from the
    ///    file's path.
    repo_path: RepoPathBuf,
}

/// Applies `run_tool()` to the inputs and stores the resulting file content.
///
/// Returns a map describing the subset of `tool_inputs` that resulted in
/// changed file content. Failures when handling an input will cause it to be
/// omitted from the return value, which is indistinguishable from succeeding
/// with no changes.
/// TODO: Better error handling so we can tell the user what went wrong with
/// each failed input.
fn fix_file_ids<'a>(
    store: &Store,
    tools_config: &ToolsConfig,
    tool_inputs: &'a HashSet<ToolInput>,
) -> BackendResult<HashMap<&'a ToolInput, FileId>> {
    let (updates_tx, updates_rx) = channel();
    // TODO: Switch to futures, or document the decision not to. We don't need
    // threads unless the threads will be doing more than waiting for pipes.
    tool_inputs.into_par_iter().try_for_each_init(
        || updates_tx.clone(),
        |updates_tx, tool_input| -> Result<(), BackendError> {
            // The first matching tool gets its input from the committed file, and any
            // subsequent matching tool gets its input from the previous matching tool's
            // output. In either case, we store the input in `prev_content`, but we avoid
            // reading anything until we know at least one tool matches.
            let mut prev_content: Option<Vec<u8>> = None;
            let mut any_changes: bool = false;
            for tool_config in tools_config.tools.iter() {
                if tool_config.matcher.matches(&tool_input.repo_path) {
                    if prev_content.is_none() {
                        let mut content = vec![];
                        let mut read =
                            store.read_file(&tool_input.repo_path, &tool_input.file_id)?;
                        read.read_to_end(&mut content).unwrap();
                        prev_content = Some(content);
                    }
                    if let Ok(next_content) = run_tool(
                        &tool_config.command,
                        tool_input,
                        prev_content.as_ref().unwrap(),
                    ) {
                        if next_content != *prev_content.as_ref().unwrap() {
                            prev_content = Some(next_content);
                            any_changes = true;
                        }
                    }
                }
            }
            if any_changes {
                // If there were multiple matching tools, the last one might have produced the
                // same file content that we started with, which could mean we don't need to
                // rewrite a commit. Rather than keep the original file content around for
                // comparison, we just rely on the resulting FileID being the same so that we
                // don't rewrite the commit. That is not expected to be a common scenario, so
                // optimizing it is not currently important.
                let new_file_id = store.write_file(
                    &tool_input.repo_path,
                    &mut prev_content.as_ref().unwrap().as_slice(),
                )?;
                updates_tx.send((tool_input, new_file_id)).unwrap();
            }
            Ok(())
        },
    )?;
    drop(updates_tx);
    let mut result = HashMap::new();
    while let Ok((tool_input, new_file_id)) = updates_rx.recv() {
        result.insert(tool_input, new_file_id);
    }
    Ok(result)
}

/// Runs the `tool_command` to fix the given file content.
///
/// The `old_content` is assumed to be that of the `tool_input`'s `FileId`, but
/// this is not verified.
///
/// Returns the new file content, whose value will be the same as `old_content`
/// unless the command introduced changes. Returns `None` if there were any
/// failures when starting, stopping, or communicating with the subprocess.
fn run_tool(
    tool_command: &CommandNameAndArgs,
    tool_input: &ToolInput,
    old_content: &[u8],
) -> Result<Vec<u8>, ()> {
    // TODO: Pipe stderr so we can tell the user which commit, file, and tool it is
    // associated with.
    let mut vars: HashMap<&str, &str> = HashMap::new();
    vars.insert("path", tool_input.repo_path.as_internal_file_string());
    let mut child = tool_command
        .to_command_with_variables(&vars)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .or(Err(()))?;
    let mut stdin = child.stdin.take().unwrap();
    let output = std::thread::scope(|s| {
        s.spawn(move || {
            stdin.write_all(old_content).ok();
        });
        Some(child.wait_with_output().or(Err(())))
    })
    .unwrap()?;
    if output.status.success() {
        Ok(output.stdout)
    } else {
        Err(())
    }
}

/// Represents an entry in the `fix.tools` config table.
struct ToolConfig {
    /// The command that will be run to fix a matching file.
    command: CommandNameAndArgs,
    /// The matcher that determines if this tool matches a file.
    matcher: Box<dyn Matcher>,
    // TODO: Store the `name` field here and print it with the command's stderr, to clearly
    // associate any errors/warnings with the tool and its configuration entry.
}

/// Represents the `fix.tools` config table.
struct ToolsConfig {
    /// Some tools, stored in the order they will be executed if more than one
    /// of them matches the same file.
    tools: Vec<ToolConfig>,
}

/// Parses the `fix.tools` config table.
///
/// Parses the deprecated `fix.tool-command` config as if it was the first entry
/// in `fix.tools`.
///
/// Fails if any of the commands or patterns are obviously unusable, but does
/// not check for issues that might still occur later like missing executables.
/// This is a place where we could fail earlier in some cases, though.
fn get_tools_config(
    workspace_command: &WorkspaceCommandHelper,
    config: &config::Config,
) -> Result<ToolsConfig, CommandError> {
    let mut tools_config = ToolsConfig { tools: Vec::new() };
    if let Ok(tool_command) = config.get("fix.tool-command") {
        // This doesn't change the displayed indices of the `fix.tools` definitions, and
        // doesn't have a `name` that could conflict with them. That would matter more
        // if we already had better error handling that made use of the `name`.
        tools_config.tools.push(ToolConfig {
            command: tool_command,
            matcher: Box::new(EverythingMatcher),
        });
    }
    if let Ok(tool_config_array) = config.get_array("fix.tools") {
        let mut tool_name_first_seen_index = HashMap::new();
        let mut tools: Vec<ToolConfig> = tool_config_array
            .into_iter()
            .enumerate()
            .map(|(index, val)| -> Result<ToolConfig, CommandError> {
                let table = val.into_table()?;
                let name = table
                    .get("name")
                    .ok_or(config::ConfigError::Message(format!(
                        "`fix.tools` entry at index {} is missing required field `name`.",
                        index
                    )))?
                    .to_string();
                if let Some(first_index) = tool_name_first_seen_index.insert(name.clone(), index) {
                    return Err(config::ConfigError::Message(format!(
                        "`fix.tools` entries at indices {} and {} have the same `name`: {}",
                        first_index, index, name
                    ))
                    .into());
                }
                let patterns: Vec<String> = table
                    .get("patterns")
                    .ok_or(config::ConfigError::Message(format!(
                        "`fix.tools` entry `{}` at index {} is missing required field `patterns`.",
                        name, index
                    )))?
                    .clone()
                    .into_array()?
                    .into_iter()
                    .map(|value| -> Result<String, ConfigError> { value.into_string() })
                    .try_collect()?;
                if patterns.is_empty() {
                    Err(config::ConfigError::Message(format!(
                        "`fix.tools` entry `{}` at index {} must have at least one pattern.",
                        name, index
                    ))
                    .into())
                } else {
                    Ok(ToolConfig {
                        command: CommandNameAndArgs::Vec(
                            NonEmptyCommandArgsVec::try_from(
                                table
                                    .get("command")
                                    .ok_or(config::ConfigError::Message(format!(
                                        "`fix.tools` entry `{}` at index {} is missing required \
                                         field `command`.",
                                        name, index
                                    )))?
                                    .clone()
                                    .into_array()?
                                    .iter()
                                    .map(|value| value.to_string())
                                    .collect_vec(),
                            )
                            .map_err(|e| {
                                config::ConfigError::Message(format!(
                                    "`fix.tools` entry `{}` at index {}: {}",
                                    name, index, e
                                ))
                            })?,
                        ),
                        matcher: workspace_command
                            .parse_union_filesets(&patterns)?
                            .to_matcher(),
                    })
                }
            })
            .try_collect()?;
        tools_config.tools.append(&mut tools);
    }
    if tools_config.tools.is_empty() {
        // TODO: This is not a useful message when one or both fields are present but
        // have the wrong type. After removing `fix.tool-command`, it will be simpler to
        // propagate any errors from `config.get_array("fix.tools")`.
        Err(config::ConfigError::Message(
            "At least one entry of `fix.tools` or `fix.tool-command` is required.".to_string(),
        )
        .into())
    } else {
        Ok(tools_config)
    }
}
