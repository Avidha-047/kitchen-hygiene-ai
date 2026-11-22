"""Bayesian network for "was soap actually used?" (sir's question: same rubbing, no soap).

Structure (course Unit 8):

            SoapUsed
          /    |     \\
   Dispenser  Lather  LongRub(>=20 s)

The three evidence nodes are conditionally independent given SoapUsed, so the joint is
P(S, D, L, R) = P(S) P(D|S) P(L|S) P(R|S), and the posterior follows from Bayes' theorem
(Unit 7) by enumerating S in {true, false}. Unobserved evidence (e.g. hands hidden, so lather
unknown) is simply summed out, which for a leaf node means leaving it out of the product.
"""


def p_soap(cfg, dispenser=None, lather=None, longrub=None):
    """Posterior P(SoapUsed = true | observed evidence). Pass None for evidence that was not observed."""
    cpts = [(dispenser, cfg.p_dispenser_given_soap),
            (lather, cfg.p_lather_given_soap),
            (longrub, cfg.p_longrub_given_soap)]
    num_true = cfg.p_soap_prior
    num_false = 1.0 - cfg.p_soap_prior
    for value, (p_if_soap, p_if_not) in cpts:
        if value is None:
            continue
        num_true *= p_if_soap if value else (1 - p_if_soap)
        num_false *= p_if_not if value else (1 - p_if_not)
    return num_true / (num_true + num_false)


def explain(cfg, dispenser=None, lather=None, longrub=None):
    parts = []
    for name, v in (("dispenser touched", dispenser), ("lather seen", lather), ("rubbed >= 20 s", longrub)):
        parts.append(f"{name}: {'yes' if v else 'no' if v is not None else 'not observed'}")
    return "; ".join(parts) + f" -> P(soap) = {p_soap(cfg, dispenser, lather, longrub):.2f}"
