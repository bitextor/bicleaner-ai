# Task Tracking System

<!-- SCOPE: Task tracking system workflow and rules ONLY. Contains task lifecycle, naming conventions, and integration rules. -->
<!-- DO NOT add here: actual task details -> task files, kanban status -> kanban_board.md, implementation guides -> guides/ -->

---

## Overview

This folder contains the project's task management system, organizing all development work into trackable units with clear status progression.

### Folder Structure

```
docs/tasks/
+-- README.md           # This file - Task tracking workflow and rules
+-- kanban_board.md     # Live navigation to active tasks in Linear
```

> [!NOTE]

> All task tracking (Epics, User Stories, tasks) is handled in Linear. Linear is the single source of truth.

**Live Navigation**: [Kanban Board](kanban_board.md)

---

## Core Concepts

### Task Lifecycle

**Workflow:**
```
Backlog/Postponed -> Todo -> In Progress -> To Review -> Done
                                              |
                                         To Rework -> (back to In Progress)
```

**Statuses:**
- **Backlog:** New tasks requiring estimation and approval
- **Postponed:** Deferred tasks for future iterations
- **Todo:** Approved tasks ready for development
- **In Progress:** Currently being developed
- **To Review:** Awaiting code review and validation
- **To Rework:** Needs fixes after review
- **Done:** Completed, reviewed, tested, approved

**Manual Statuses** (not in workflow): Canceled, Duplicate

### Epic Structure

**Organization in Linear:**
- **Epic** = Linear Project (contains all User Stories and tasks for epic)
- **User Story** = Linear Issue with `label: user-story` and `parentId: null`
- **Task** = Linear Issue with `parentId: <UserStoryId>`

**Epic Fields:** Name, description, start date, target date, project lead
**User Story Format:** "As a... I want... So that..." + Given-When-Then acceptance criteria
**Task Format:** Context, requirements, acceptance criteria, implementation notes

### Foundation-First Execution Order

**Critical Rule**: Foundation tasks are executed BEFORE consumer tasks (for testability).

**Definitions**:
- **Foundation** = Database, Repository, core services
- **Consumer** = API endpoints, Frontend components that USE foundation

**Rationale**: Each layer is testable when built (can't test API without working DB).

**Example**:
```
CORRECT EXECUTION ORDER:
  Task 1: Database schema + Repository (foundation)
  Task 2: Service layer with business logic
  Task 3: API endpoint (consumer)
  Task 4: Frontend dashboard (consumer)

WRONG (can't test):
  Task 1: Frontend dashboard calls /api/users
  Task 2: API endpoint (no DB to test against)
```

> **Note:** Consumer-First is for API/interface DESIGN (think from consumer perspective), Foundation-First is for EXECUTION ORDER (build testable foundation first).

---

## Critical Rules

### Rule 1: Linear Integration (MCP Methods ONLY)

**CRITICAL**: Use ONLY `mcp__linear-server__*` methods for all Linear operations.

**PROHIBITED**:
- `gh` command (GitHub CLI)
- GitHub API calls
- Direct Linear API calls
- Manual task creation in Linear UI (for automated workflows)

**Rationale**:
- MCP Linear provides type-safe, validated operations
- Prevents data inconsistencies
- Ensures proper error handling

**See**: [Linear MCP Methods Reference](#linear-mcp-methods-reference) below

---

### Rule 2: Integration Rules

#### Tests

**Rule**: Tests are created ONLY in the final Story task (Story Finalizer test task).

**NEVER** create:
- Separate test tasks during implementation
- Tests in implementation tasks (implementation tasks focus on feature code only)

**Process**:
1. Implementation tasks (1-6 tasks) -> To Review -> Done
2. ln-500-story-quality-gate Pass 1 -> Manual testing
3. ln-510-test-planner -> Creates Story Finalizer test task
4. ln-404-test-executor -> Implements all tests (E2E, Integration, Unit)

**Rationale**: Atomic testing strategy, prevents test duplication, ensures comprehensive coverage.

#### Documentation

**Rule**: Documentation is ALWAYS integrated in feature tasks (same task as implementation).

**NEVER** create:
- Separate documentation tasks
- "Update README" tasks
- "Write API docs" tasks

**Process**: Implementation task includes both code AND documentation updates in Definition of Done.

**Rationale**: Ensures documentation stays in sync with code, prevents documentation debt.

---

### Rule 3: Story-Level Test Strategy

**Value-Based Testing**: Test only scenarios with Priority >=15 (calculated by Impact x Likelihood).

**Test Limits per Story**:

| Test Type | Min | Max | Purpose |
|-----------|-----|-----|---------|
| **E2E Tests** | 2 | 5 | End-to-end user workflows (Priority >=15) |
| **Integration Tests** | 3 | 8 | Component interactions, external APIs |
| **Unit Tests** | 5 | 15 | Business logic, edge cases |
| **Total** | 10 | 28 | Complete Story coverage |

**Example**:
```
Story: User Authentication
- E2E: 3 tests (login success, login failure, session expiry)
- Integration: 5 tests (OAuth flow, token refresh, database session storage, Redis cache, logout)
- Unit: 8 tests (password validation, email validation, token generation, permission checks, etc.)
Total: 16 tests (within 10-28 range)
```

**Reference**: [Testing Strategy](../reference/guides/testing-strategy.md) for bicleaner-ai specific patterns.

---

### Rule 4: Context Budget Rule

- [ ] **CRITICAL: Minimize context pollution in kanban_board.md**

**Rule:** [kanban_board.md](./kanban_board.md) contains ONLY links and titles - no descriptions, no implementation notes.

**Board Structure:**

Single hierarchical view: **Status -> Epic -> User Story -> Tasks**

**Sections:**
1. **Work in Progress** - Hierarchical task tracking (Backlog -> Todo -> In Progress -> To Review -> To Rework -> Done -> Postponed)
2. **Epics Overview** - Portfolio-level status (Active + Completed epics)

**Format Rules:**

**User Story:**
- Format: `[TEAM_KEY]-XX: USYYY Title` + optional `APPROVED`
- 2-space indent from Epic
- Always shows parent epic context
- Can exist without tasks ONLY in Backlog status (with note: `_(tasks not created yet)_`)

**Task:**
- Format: `  - [TEAM_KEY]-XX: EPYY_ZZ Title` (2-space indent + dash)
- 4-space total indent (2-space base from Story + 2-space for dash)
- Always nested under parent User Story
- Cannot exist without parent story

**Epic Grouping:**
- Each status section grouped by: `**Epic N: Epic Name**` (bold header)
- Stories listed under epic with 2-space indent
- Tasks listed under stories with 4-space indent (2-space base + 2-space for dash)

**Status-Specific Limits:**
- **Backlog:** All stories (tasks optional, use `_(tasks not created yet)_` if none)
- **Todo/In Progress/To Review/To Rework:** All stories with all tasks
- **Done:** Last 5 stories ONLY (no tasks - removed from tracking after completion)
- **Postponed:** Stories ONLY (no tasks - tasks created when work resumes)

**Rationale:**
- Single view eliminates need to "jump" between sections
- Natural hierarchy matches mental model: Status -> Epic -> Story -> Task
- Story context always visible with its tasks
- Reduced cognitive load: one structure, not three
- Minimize context size for AI agents
- Fast navigation at all levels (status/epic/story/task)

---

## Task Workflow

### Planning Guidelines

**Optimal Task Size**: 3-5 hours per task

**Task Granularity**:
- Too small (< 2 hours): Merge with related tasks
- Too large (> 8 hours): Split into subtasks
- Sweet spot (3-5 hours): Maximum productivity, clear acceptance criteria

**Story Limits**:
- Implementation tasks: 1-6 tasks per Story
- Test task: 1 Story Finalizer test task (created after implementation)
- Total: Max 7 tasks per Story

### Workflow Skills

| Category | Skill | Purpose |
|----------|-------|---------|
| **Planning** | ln-210-epic-coordinator | Decompose scope -> 3-7 Epics |
| | ln-220-story-coordinator | Decompose Epic -> 5-10 Stories (with Phase 3 Library Research) |
| | ln-300-task-coordinator | Decompose Story -> 1-6 Implementation Tasks |
| | ln-510-test-planner | Plan Story Finalizer test task (after manual testing) |
| **Validation** | ln-310-story-validator | Auto-fix Stories/Tasks -> Approve (Backlog -> Todo) |
| **Execution** | ln-400-story-executor | Orchestrate Story execution (delegates to ln-401/ln-404/ln-402) |
| | ln-401-task-executor | Execute implementation tasks (Todo -> In Progress -> To Review) |
| | ln-404-test-executor | Execute Story Finalizer test tasks (11 sections) |
| | ln-402-task-reviewer | Review tasks (To Review -> Done/Rework) |
| | ln-403-task-rework | Fix tasks after review (To Rework -> To Review) |
| **Quality** | ln-500-story-quality-gate | Two-pass review (Code Quality -> Regression -> Manual Testing) |
| | ln-501-code-quality-checker | Analyze code for DRY/KISS/YAGNI violations |
| | ln-502-regression-checker | Run existing test suite |
| | ln-512-manual-tester | Perform manual testing via curl/puppeteer |
| **Documentation** | ln-111-project-docs-creator | Create project docs (requirements, architecture, specs) |
| | ln-002-best-practices-researcher | Create ADRs, guides, manuals (doc_type parameter) |

---

## Project Configuration

### Quality Commands

```bash
# Install (development)
pip install -e ".[dev]"

# Install (all features)
pip install ".[all]"

# Run tests
pytest tests/

# Run tests with coverage
pytest tests/ --cov=src/bicleaner_ai

# Type checking (not configured)
# mypy src/bicleaner_ai/

# Linting (not configured)
# ruff check src/
```

### Documentation Structure

Core documentation:
- [Architecture](../architecture.md) - Library architecture, model types, data flow
- [Tech Stack](../tech_stack.md) - Dependencies, frameworks, compatibility
- [Training](../training/README.md) - Model training guides
- [ADRs](../reference/adrs/) - Architecture Decision Records
- [Guides](../reference/guides/) - Project patterns and best practices

### Label Taxonomy

**Functional Labels**: `feature`, `bug`, `refactoring`, `documentation`, `testing`, `infrastructure`

**Type Labels**: `user-story`, `implementation-task`, `test-task`

**Status Labels** (auto-managed by Linear): `backlog`, `todo`, `in-progress`, `to-review`, `to-rework`, `done`, `canceled`

---

## Linear MCP Methods Reference

### Core Operations

**Issues**:

```python
# List issues
mcp__linear-server__list_issues(
    team="TeamName",           # Team name or ID
    state="In Progress",       # State name or ID
    assignee="me",             # User ID, name, email, or "me"
    limit=50                   # Max 250
)

# Get issue details
mcp__linear-server__get_issue(
    id="PROJ-123"              # Issue ID
)

# Create issue
mcp__linear-server__create_issue(
    team="TeamName",           # Required: Team name or ID
    title="Task title",        # Required: Issue title
    description="...",         # Markdown description
    state="Todo",              # State name or ID
    assignee="me",             # User ID, name, email, or "me"
    parentId="parent-uuid",    # For tasks (parent Story UUID)
    labels=["feature", "backend"]  # Label names or IDs
)

# Update issue
mcp__linear-server__update_issue(
    id="PROJ-123",             # Required: Issue ID
    state="Done",              # State name or ID
    description="...",         # Updated description
    assignee="me"              # Reassign
)
```

**Projects** (Epics):

```python
# List projects
mcp__linear-server__list_projects(
    team="TeamName",           # Filter by team
    state="started",           # Filter by state
    limit=50
)

# Get project
mcp__linear-server__get_project(
    query="Epic 1"             # ID or name
)

# Create project (Epic)
mcp__linear-server__create_project(
    team="TeamName",           # Required
    name="Epic 1: Auth",       # Required
    description="...",         # Epic description
    state="planned"            # Epic state
)

# Update project
mcp__linear-server__update_project(
    id="project-uuid",         # Required
    state="started",           # Update state
    description="..."          # Update description
)
```

**Teams**:

```python
# List teams
mcp__linear-server__list_teams()

# Get team
mcp__linear-server__get_team(
    query="TeamName"           # UUID, key, or name
)
```

**Labels**:

```python
# List labels
mcp__linear-server__list_issue_labels(
    team="TeamName"            # Optional: filter by team
)

# Create label
mcp__linear-server__create_issue_label(
    name="backend",            # Required
    color="#FF5733",           # Hex color
    teamId="team-uuid"         # Optional: team-specific label
)
```

**Comments**:

```python
# List comments
mcp__linear-server__list_comments(
    issueId="issue-uuid"       # Required
)

# Create comment
mcp__linear-server__create_comment(
    issueId="issue-uuid",      # Required
    body="Comment text"        # Required: Markdown
)
```

### Parameter Patterns

**Common Filters**:

```python
# By assignee
assignee="me"                  # Current user
assignee="user@example.com"    # By email
assignee="user-uuid"           # By UUID

# By state
state="Todo"                   # By name
state="state-uuid"             # By UUID

# By team
team="TeamName"                # By name
team="team-uuid"               # By UUID

# By label
label="feature"                # By name
label="label-uuid"             # By UUID
```

**Pagination**:

```python
# Limit results
limit=50                       # Default: 50, Max: 250

# Pagination
after="cursor-id"              # Start from cursor
before="cursor-id"             # End at cursor
```

**Date Filters**:

```python
# ISO-8601 date-time
createdAt="2025-01-01T00:00:00Z"

# Duration (relative)
createdAt="-P1D"               # Created in last 1 day
updatedAt="-P7D"               # Updated in last 7 days
```

### Examples

**Example 1: Get all In Progress tasks for current user**

```python
issues = mcp__linear-server__list_issues(
    team="ProjectTeam",
    state="In Progress",
    assignee="me",
    limit=100
)
```

**Example 2: Create Story (User Story)**

```python
story = mcp__linear-server__create_issue(
    team="ProjectTeam",
    title="US001 User Login",
    description="User story description...",
    state="Backlog",
    labels=["user-story", "feature"]
)
```

**Example 3: Create Task (child of Story)**

```python
task = mcp__linear-server__create_issue(
    team="ProjectTeam",
    title="EP1_01 Implement JWT tokens",
    description="Task description...",
    state="Todo",
    parentId="story-uuid",      # Link to parent Story
    labels=["implementation-task", "backend"]
)
```

**Example 4: Move task to Done**

```python
mcp__linear-server__update_issue(
    id="PROJ-123",
    state="Done"
)
```

**Example 5: Create Epic (Project)**

```python
epic = mcp__linear-server__create_project(
    team="ProjectTeam",
    name="Epic 1: Authentication System",
    description="Epic description...",
    state="planned"
)
```

---

## Maintenance

**Update Triggers**:
- When adding new workflow skills
- When changing task lifecycle statuses
- When updating Critical Rules
- When modifying Linear integration patterns
- When changing test strategy limits

**Verification**:
- All Linear MCP method examples are valid
- Workflow skills table matches available skills
- Critical Rules align with current development principles
- Test limits match risk-based testing guide

**Last Updated**: 2026-01-22
