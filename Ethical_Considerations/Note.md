## Ethical Considerations

### **Vulnerability Disclosure**

Our research examines adversarial prompting techniques for evaluating the robustness of language-model–based content moderation systems, including dedicated moderation APIs, general-purpose LLM APIs configured for moderation via prompting, and locally deployed LLM moderation components. All experiments were performed exclusively through publicly documented interfaces or within locally controlled environments. We did not access internal systems, circumvent protections, or interact with any user data.

The issues identified in this work represent behavioral limitations of model-based moderation mechanisms rather than vulnerabilities in deployed software or infrastructure, and therefore do not fall under traditional vulnerability-disclosure processes. No harm was caused to vendors or users at any point during our study.

---

### **Human Subjects Research**

As noted earlier, this study received IRB approval, and all procedures related to informed consent, participant protections, and data handling were reviewed and implemented accordingly. This section provides additional clarification to align with standard human-subjects research requirements.

Our study involved participant judgments on brief text excerpts—some containing sensitive language—used solely to evaluate evasion of LLM-based moderation mechanisms. Upon review, the IRB classified the study as *minimal risk*, and participants were free to withdraw at any time.

No personally identifiable information (PII) was collected. All responses were anonymized upon submission and analyzed only in aggregate form. Data were stored securely with restricted access, ensuring that participants could not be re-identified. These procedures ensure compliance with ethical standards for human-subjects research and the mitigation of any potential harms.

