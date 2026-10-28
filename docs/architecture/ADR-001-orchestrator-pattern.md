# ADR-001: Orchestrator Pattern for Chart Analysis System

**Status:** Accepted  
**Date:** 2025-09-01  
**Deciders:** Architecture Team  
**Technical Story:** Initial System Design

## Context and Problem Statement

The Chart Analysis System requires a robust, scalable, and maintainable architecture to process diverse chart types, integrate multiple computer vision and OCR components, and support future expansion. A monolithic approach led to tight coupling, reduced flexibility, and increased complexity in managing different processing flows for each chart type.

## Decision Drivers

*   **Maintainability:** Ease of understanding, modifying, and extending the system.
*   **Scalability:** Ability to add new chart types or processing steps without significant refactoring.
*   **Flexibility:** Decoupling of components to allow independent development and updates.
*   **Testability:** Facilitate isolated testing of individual processing units.

## Considered Options

1.  **Monolithic Architecture:** A single, tightly coupled application handling all logic.
2.  **Microservices Architecture:** Decompose the system into independent, deployable services.
3.  **Service-Oriented Architecture (SOA) with Orchestrator Pattern:** A layered architecture where an orchestrator coordinates shared services and specialized handlers.

## Decision Outcome

**Chosen Option:** Option 3 - Service-Oriented Architecture (SOA) with Orchestrator Pattern

**Rationale:**
The SOA with an orchestrator pattern provides a balanced approach, offering better modularity and flexibility than a monolith without the overhead and complexity of a full microservices implementation. The `ChartAnalysisOrchestrator` acts as a central routing system, delegating tasks to specialized handlers and shared services, ensuring clear separation of concerns and promoting reusability of core algorithms.

### Consequences

**Positive:**
*   **Improved Modularity:** Clear separation into layers (Orchestrator, Handler, Service, Core, Extractor) enhances code organization.
*   **Easier Extension:** Adding new chart types primarily involves creating a new handler and registering it with the orchestrator.
*   **Enhanced Maintainability:** Components can be developed, tested, and updated independently.
*   **Reusability:** Core algorithms (e.g., `ModularBaselineDetector`, LYLAA) are centralized in the Core Layer and composed by handlers.

**Negative:**
*   **Increased Initial Complexity:** Requires careful design of interfaces and communication protocols between layers.
*   **Potential for Bottlenecks:** The orchestrator could become a single point of failure or performance bottleneck if not designed efficiently (mitigated by asynchronous processing and optimized routing).

## Implementation Details

**Code Locations:**
*   `src/ChartAnalysisOrchestrator.py` (Orchestrator Layer)
*   `src/handlers/` (Handler Layer)
*   `src/services/` (Service Layer)
*   `src/core/` (Core Layer)
*   `src/extractors/` (Extractor Layer)

**Configuration:**
*   Handler registry managed internally by `ChartAnalysisOrchestrator`.

**Testing:**
*   Unit tests for individual handlers and services.
*   Integration tests for the orchestrator's routing logic.

## Validation

**Metrics:**
*   Time to integrate a new chart type (target: < 1 day).
*   Number of code changes required for a new feature (target: localized to relevant layer).

**Evidence:**
*   Successful integration of multiple chart types (bar, line, scatter, box, histogram).
*   Modular updates to core algorithms without affecting other handlers.
