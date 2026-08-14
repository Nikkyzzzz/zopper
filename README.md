IntelliSynth Project Demo / Client Presentation Speech
Introduction

Good morning everyone.

Today, I will be presenting IntelliSynth, an AI-powered document intelligence platform designed to transform complex insurance documents into a structured, queryable, and temporally-aware knowledge graph.

In the insurance industry, critical information is often spread across multiple documents, including policy documents, riders, endorsements, amendments, and coverage updates. As these documents evolve over time, determining exactly what policy was in effect on a specific date becomes extremely challenging.

IntelliSynth addresses this problem by leveraging Large Language Models, graph technology, and temporal reasoning to automatically extract information, track changes over time, and provide accurate point-in-time policy intelligence.

Business Problem

Let's first understand the challenge.

In a typical insurance ecosystem, a policy does not exist as a single document.

Instead, we have:

Base policy documents
Riders
Amendments
Endorsements
Coverage updates

Each of these modifies the policy over time.

For example:

A rider may change the copay amount.
An amendment may introduce a new coverage rule.
An endorsement may override an existing exclusion.

As a result, analysts, compliance teams, and claims processors often spend hours manually reviewing documents to answer simple questions such as:

What was the active policy on a specific date?
Which document introduced a particular rule?
What changed between two policy versions?

This manual process is slow, expensive, and prone to human error.

Solution Overview

IntelliSynth converts unstructured insurance documents into a centralized temporal knowledge graph.

The platform performs:

AI-powered document extraction.
Entity identification and normalization.
Temporal relationship construction.
Graph validation.
Graph persistence.
Intelligent consumption through APIs, reports, visualizations, and point-in-time queries.

The end result is a single source of truth for policy intelligence.

End-to-End Workflow

IntelliSynth follows a seven-stage workflow.

Create Job
    ↓
Upload Documents
    ↓
Process Job
    ↓
Extraction
    ↓
Stitching
    ↓
Validation
    ↓
Persistence
    ↓
Consumption


Let's walk through each of these stages.

Stage 1: Create Job

The process begins with creating a job.

A job acts as an isolated workspace where all documents, processing results, validation reports, and graph snapshots are stored.

This provides:

Data isolation
Auditability
Traceability
Multi-user support
Repeatable processing

Each job represents one insurance plan's processing lifecycle.

Stage 2: Process Job

After document upload, users initiate processing.

This acts as the orchestration layer.

The system automatically executes:

Extraction
→ Stitching
→ Validation
→ Persistence


without requiring manual intervention.

This creates a streamlined and automated document intelligence workflow.

Stage 3: Extraction Phase

This is where the AI layer comes into action.

The extractor module utilizes Google Gemini to read and understand insurance documents.

Supported documents include:

Policy documents
Amendments
Riders
Endorsements
Coverage files

The system identifies and extracts:

Plans
Drugs
Tiers
Copays
Prior authorization rules
Quantity limits
Step therapy rules
Effective dates
Amendment information

For example:

A document stating:

"Metformin 500mg belongs to Tier 2 with a $20 copay effective January 2024"

is automatically converted into structured JSON.

To ensure reliability, every LLM response passes through Pydantic validation schemas.

This validation layer guarantees:

Schema consistency
Correct data types
Missing field detection
Normalized structure

Additionally, all extraction results are cached.

This means documents are processed by Gemini only once, significantly reducing cost and improving performance.

Stage 4: Stitching Phase

Once individual document extractions are complete, the stitching engine takes over.

This is one of the most important stages of IntelliSynth.

Insurance information is often distributed across multiple documents.

For example:

Policy:
Copay = $10

Rider:
Copay = $20

Amendment:
Copay = $30


The stitcher reconciles all these changes and builds a single timeline.

Key responsibilities include:

Entity Resolution

Determining that references such as:

Metformin
Metformin 500mg

represent the same business entity.

Relationship Creation

The system generates graph relationships such as:

HAS_COPAY
HAS_TIER
MODIFIED_BY
BELONGS_TO
OVERRIDES
SUPERSEDES
Temporal Modeling

The system builds validity windows.

For example:

Jan 2024 - Aug 2024
Copay = $20

Sep 2024 onwards
Copay = $30


This enables historical policy reconstruction.

Stage 5: Validation Phase

Before publishing the graph, IntelliSynth validates its integrity.

Validation occurs across three dimensions.

Structural Validation

Checks:

Missing nodes
Broken relationships
Duplicate IDs
Orphan entities
Temporal Validation

Checks:

Invalid intervals
Overlapping date ranges
Timeline inconsistencies
Business Validation

Checks:

Single plan enforcement
Valid relationships
Required attributes
Rule consistency

Only graphs that pass all validation rules are allowed to proceed.

This ensures downstream consumers always receive trustworthy data.

Stage 6: Persistence Phase

After validation succeeds, the generated assets are persisted.

This includes:

Raw Extractions

Original AI extraction outputs are securely stored.

Benefits:

Rebuild support
Auditing
Traceability
Graph Storage

Internally, IntelliSynth uses:

networkx.MultiDiGraph


which naturally models:

Complex relationships
Temporal edges
Version lineage
Entity hierarchies
Graph Snapshots

Every successful build generates a new graph snapshot.

Benefits include:

Historical comparison
Rollback capability
Version management
Audit reporting
Stage 7: Consumption Phase

This is where business users derive value from the platform.

IntelliSynth offers multiple consumption methods.

Analyst Queries

Users can ask:

rule_on(
    graph,
    "Metformin 500mg",
    "copay",
    "2024-08-01"
)


The platform returns the exact value active on that date.

Point-in-Time Policy Reconstruction

Users can execute:

plan_as_of(graph, "2024-08-01")


This reconstructs the entire policy exactly as it existed on the selected date.

This capability is one of IntelliSynth's strongest differentiators.

Analyst Tables

The system generates business-friendly denormalized tables that show:

Drug
Tier
Copay
Source document
Effective rule

This enables easy reporting and analysis.

Excel Export

Users can export results directly into Excel workbooks with multiple sheets for different entity types.

Interactive Graph Visualization

A graphical interface enables:

Relationship exploration
Policy lineage tracking
Temporal analysis
Interactive navigation

This significantly simplifies understanding complex document structures.

API Integration

The platform exposes REST APIs that allow integration with:

Claims systems
Underwriting platforms
Compliance tools
Internal dashboards
Future AI assistants
Key Innovation

The real innovation of IntelliSynth is not just document extraction.

Many solutions can extract information.

What differentiates IntelliSynth is its ability to answer:

"What was true on a specific date?"

By combining:

LLM-based extraction
Knowledge graphs
Version tracking
Temporal intelligence

the platform can reconstruct historical policy states with complete traceability.

Business Benefits

IntelliSynth delivers significant business value:

✅ Reduced manual document review

✅ Faster policy analysis

✅ Improved claim processing accuracy

✅ Better compliance and audit readiness

✅ Complete policy lineage tracking

✅ Reliable point-in-time decision making

✅ Enterprise-scale knowledge management

Closing

To summarize, IntelliSynth transforms disconnected insurance documents into a validated, temporally-aware knowledge graph that enables users to understand not only what the policy says today, but also what the policy said at any point in time.

By combining AI-powered extraction, intelligent graph construction, temporal reasoning, and rich consumption capabilities, IntelliSynth creates a powerful foundation for insurance intelligence, compliance, claims processing, and decision support.

Thank you. I would be happy to take any questions.
