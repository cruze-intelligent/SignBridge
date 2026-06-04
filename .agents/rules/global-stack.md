---
trigger: always_on
---

# Global Tech Stack Constraints

- [cite_start]**Architecture:** Decoupled Client-Server model[cite: 85, 86].
- [cite_start]**No Heavy Frontend Frameworks:** The client-side user interface MUST be built strictly using a Vanilla HTML/CSS/JavaScript stack[cite: 91]. Do not install React, Vue, Angular, or any heavy bundlers unless explicitly requested.
- [cite_start]**Backend Language:** The server node MUST be written in Python[cite: 98].
- **Formatting & Linting:** Enforce strict PEP 8 for Python and standard Prettier formatting for HTML/CSS/JS.
- [cite_start]**Mobile-First:** All UI generation must prioritize a mobile-first responsive design, optimized specifically for smartphones[cite: 307].