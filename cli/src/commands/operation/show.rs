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

use itertools::Itertools;
use jj_lib::op_walk;

use super::diff::show_op_diff;
use crate::cli_util::{CommandHelper, LogContentFormat};
use crate::command_error::{user_error, CommandError};
use crate::diff_util::DiffFormatArgs;
use crate::operation_templater::OperationTemplateLanguage;
use crate::ui::Ui;

/// Show changes to the repository in an operation
#[derive(clap::Args, Clone, Debug)]
pub struct OperationShowArgs {
    /// Show repository changes in this operation, compared to its parent(s)
    #[arg(visible_alias = "op", default_value = "@")]
    operation: String,
    /// Don't show the graph, show a flat list of modified changes
    #[arg(long)]
    no_graph: bool,
    /// Show patch of modifications to changes
    ///
    /// If the previous version has different parents, it will be temporarily
    /// rebased to the parents of the new version, so the diff is not
    /// contaminated by unrelated changes.
    #[arg(long, short = 'p')]
    patch: bool,
    #[command(flatten)]
    diff_format: DiffFormatArgs,
}

pub fn cmd_op_show(
    ui: &mut Ui,
    command: &CommandHelper,
    args: &OperationShowArgs,
) -> Result<(), CommandError> {
    let workspace_command = command.workspace_helper(ui)?;
    let repo = workspace_command.repo();
    let repo_loader = &repo.loader();
    let head_op_str = &command.global_args().at_operation;
    let head_ops = if head_op_str == "@" {
        // If multiple head ops can't be resolved without merging, let the
        // current op be empty. Beware that resolve_op_for_load() will eliminate
        // redundant heads whereas get_current_head_ops() won't.
        let current_op = op_walk::resolve_op_for_load(repo_loader, head_op_str).ok();
        if let Some(op) = current_op {
            vec![op]
        } else {
            op_walk::get_current_head_ops(
                repo_loader.op_store(),
                repo_loader.op_heads_store().as_ref(),
            )?
        }
    } else {
        vec![op_walk::resolve_op_for_load(repo_loader, head_op_str)?]
    };
    let current_op_id = match &*head_ops {
        [op] => Some(op.id()),
        _ => None,
    };
    let op = op_walk::resolve_op_for_load(repo_loader, &args.operation)?;
    let parents: Vec<_> = op.parents().try_collect()?;
    if parents.is_empty() {
        return Err(user_error("Cannot show the root operation"));
    }
    let parent_op = repo_loader.merge_operations(command.settings(), parents, None)?;
    let parent_repo = repo_loader.load_at(&parent_op)?;
    let repo = repo_loader.load_at(&op)?;

    // Create a new transaction beginning from `repo`.
    let mut workspace_command =
        command.for_loaded_repo(ui, command.load_workspace()?, repo.clone())?;
    let tx = workspace_command.start_transaction();

    let with_content_format = LogContentFormat::new(ui, command.settings())?;

    // TODO: Should we make this customizable via clap arg?
    let template;
    {
        let language = OperationTemplateLanguage::new(
            repo_loader.op_store().root_operation_id(),
            current_op_id,
            command.operation_template_extensions(),
        );
        let text = command.settings().config().get_string("templates.op_log")?;
        template = command
            .parse_template(
                ui,
                &language,
                &text,
                OperationTemplateLanguage::wrap_operation,
            )?
            .labeled("op_log");
    }

    ui.request_pager();
    template.format(&op, ui.stdout_formatter().as_mut())?;

    show_op_diff(
        ui,
        command,
        tx,
        &parent_repo,
        &repo,
        !args.no_graph,
        &with_content_format,
        &args.diff_format,
        args.patch,
    )
}
