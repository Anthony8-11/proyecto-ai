import type { SimStep } from "../hooks/useSimulation";

interface Props { step: SimStep | null; }

const SECTORS = [
  { label: "Industrial", abbr: "IND", short: "S0" },
  { label: "Res. Norte", abbr: "RES", short: "S1" },
  { label: "Comercial",  abbr: "COM", short: "S2" },
  { label: "Res. Sur",   abbr: "SUR", short: "S3" },
  { label: "Planta",     abbr: "PLT", short: "S4" },
];

const COND_COLOR = ["#22c55e", "#f59e0b", "#ef4444"];
const COND_LABEL = ["Estable", "Sobrecarga", "Fallo de red"];
const COND_BG    = ["#052e1688", "#1c100388", "#2d050588"];

// Node positions inside SVG viewBox "0 0 1000 220"
const NODE_X = [96, 288, 480, 672, 864];
const NODE_Y = 110;
const R = 42;

function FlowLine({
  idx, isActive, condColor,
}: { idx: number; isActive: boolean; condColor: string }) {
  const x1 = NODE_X[idx], x2 = NODE_X[idx + 1];
  const pathId = `fp-${idx}`;
  const dotColor = isActive ? condColor : "#1e4068";
  const lineColor = isActive ? condColor + "55" : "#162840";

  return (
    <g>
      {/* Base line */}
      <line x1={x1} y1={NODE_Y} x2={x2} y2={NODE_Y}
        stroke={lineColor} strokeWidth="3" strokeLinecap="round"/>

      {/* Path for animateMotion */}
      <path id={pathId} d={`M${x1},${NODE_Y} L${x2},${NODE_Y}`} fill="none" stroke="none"/>

      {/* 3 flowing dots */}
      {[0, 0.65, 1.3].map((offset, i) => (
        <circle key={i} r="3.5" fill={dotColor} opacity={isActive ? 0.85 : 0.35}>
          <animateMotion dur="1.8s" repeatCount="indefinite" begin={`${-offset}s`}>
            <mpath href={`#${pathId}`}/>
          </animateMotion>
        </circle>
      ))}
    </g>
  );
}

function SectorNode({
  i, pos, cond,
}: { i: number; pos: number | null; cond: number }) {
  const x = NODE_X[i];
  const isAgent = pos === i;
  const color = isAgent ? COND_COLOR[cond] : "#1e4068";
  const glow  = isAgent ? COND_BG[cond] : "transparent";
  const sector = SECTORS[i];

  return (
    <g>
      {/* Animated outer ring (only for agent) */}
      {isAgent && (
        <circle cx={x} cy={NODE_Y} r={R + 18} fill="none"
          stroke={COND_COLOR[cond]} strokeWidth="1.5">
          <animate attributeName="opacity" values="0.7;0.1;0.7" dur="2s" repeatCount="indefinite"/>
          <animate attributeName="r" values={`${R+14};${R+26};${R+14}`} dur="2s" repeatCount="indefinite"/>
        </circle>
      )}

      {/* Second ring for critical (extra urgency) */}
      {isAgent && cond === 2 && (
        <circle cx={x} cy={NODE_Y} r={R + 32} fill="none"
          stroke={COND_COLOR[cond]} strokeWidth="1">
          <animate attributeName="opacity" values="0.4;0;0.4" dur="1.2s" repeatCount="indefinite"/>
          <animate attributeName="r" values={`${R+28};${R+44};${R+28}`} dur="1.2s" repeatCount="indefinite"/>
        </circle>
      )}

      {/* Glow fill */}
      {isAgent && (
        <circle cx={x} cy={NODE_Y} r={R + 6} fill={glow}/>
      )}

      {/* Main circle */}
      <circle cx={x} cy={NODE_Y} r={R}
        fill="#060e1c"
        stroke={color}
        strokeWidth={isAgent ? 2.5 : 1.5}/>

      {/* Sector abbreviation text */}
      <text x={x} y={isAgent ? NODE_Y - 7 : NODE_Y + 2}
        textAnchor="middle" dominantBaseline="middle"
        fill={isAgent ? color : "#2a4060"}
        fontSize={isAgent ? "13" : "11"}
        fontWeight="800"
        fontFamily="'JetBrains Mono','Fira Code','Cascadia Code',monospace"
        style={{ userSelect: "none" }}>
        {sector.abbr}
      </text>

      {/* Agent indicator — small filled dot */}
      {isAgent && (
        <circle cx={x} cy={NODE_Y + 15} r={4}
          fill={COND_COLOR[cond]}
          opacity={0.9}>
          <animate attributeName="opacity" values="0.9;0.4;0.9" dur="1.2s" repeatCount="indefinite"/>
        </circle>
      )}

      {/* Condition label under agent */}
      {isAgent && (
        <text x={x} y={NODE_Y + R + 20}
          textAnchor="middle"
          fill={COND_COLOR[cond]}
          fontSize="10.5"
          fontWeight="700">
          {COND_LABEL[cond].toUpperCase()}
        </text>
      )}
      {!isAgent && (
        <text x={x} y={NODE_Y + R + 20}
          textAnchor="middle"
          fill="#2a4060"
          fontSize="10"
          fontWeight="500">
          {sector.label}
        </text>
      )}

      {/* S-index badge */}
      <rect x={x - R + 2} y={NODE_Y - R + 2}
        width="20" height="13" rx="3"
        fill="#0b1a2e" opacity="0.9"/>
      <text x={x - R + 12} y={NODE_Y - R + 8.5}
        textAnchor="middle" dominantBaseline="middle"
        fill={isAgent ? color : "#2a4060"}
        fontSize="8" fontWeight="800">
        {sector.short}
      </text>
    </g>
  );
}

export function EnvironmentMap({ step }: Props) {
  const pos  = step?.state.position ?? null;
  const cond = step?.state.env_condition ?? 0;
  const res  = step?.state.resources ?? 10;

  const capPct  = res * 10;
  const capColor = res > 6 ? "#22c55e" : res > 3 ? "#f59e0b" : "#ef4444";
  const capLabel = res > 6 ? "ALTA" : res > 3 ? "BAJA" : "CRÍTICA";
  const condColor = COND_COLOR[cond];

  return (
    <div className="card" style={{ padding: "1rem 1.5rem" }}>
      {/* Header row */}
      <div style={{
        display: "flex", alignItems: "center",
        justifyContent: "space-between", marginBottom: ".75rem",
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: ".6rem" }}>
          <span style={{
            fontSize: ".7rem", fontWeight: 800, textTransform: "uppercase",
            letterSpacing: ".1em", color: "#2a4060",
          }}>
            Red Eléctrica — Topología
          </span>
        </div>
        <div style={{ display: "flex", gap: ".5rem", alignItems: "center" }}>
          {step ? (
            <span style={{
              display: "flex", alignItems: "center", gap: 5,
              fontSize: ".68rem", fontWeight: 700, padding: "2px 10px",
              borderRadius: 6, border: `1px solid ${condColor}44`,
              background: COND_BG[cond], color: condColor,
              animation: cond > 0 ? "blink-slow 2s infinite" : "none",
            }}>
              <span style={{
                display: "inline-block", width: 6, height: 6,
                borderRadius: "50%", background: condColor, flexShrink: 0,
              }}/>
              {COND_LABEL[cond]}
            </span>
          ) : (
            <span style={{ fontSize: ".68rem", color: "#2a4060" }}>Sin simulación activa</span>
          )}
          <span style={{
            fontSize: ".65rem", padding: "2px 9px", borderRadius: 5,
            background: "#0b1a2e", border: "1px solid #1e4068", color: "#38bdf8", fontWeight: 700,
          }}>
            Centro de Control
          </span>
        </div>
      </div>

      {/* SVG network */}
      <div style={{
        background: "linear-gradient(180deg, #040c1a 0%, #060f1f 100%)",
        borderRadius: 10, border: "1px solid #162840",
        overflow: "hidden", position: "relative",
      }}>
        {/* Scanline overlay */}
        <div style={{
          position: "absolute", inset: 0, zIndex: 1,
          backgroundImage: "repeating-linear-gradient(0deg, transparent, transparent 3px, rgba(56,189,248,.015) 3px, rgba(56,189,248,.015) 4px)",
          pointerEvents: "none",
        }}/>

        <svg viewBox="0 0 1000 205" style={{ width: "100%", display: "block" }}>
          <defs>
            {/* Background grid */}
            <pattern id="grid" width="50" height="50" patternUnits="userSpaceOnUse">
              <path d="M 50 0 L 0 0 0 50" fill="none" stroke="#0c1929" strokeWidth="1"/>
            </pattern>
          </defs>

          {/* Grid background */}
          <rect width="1000" height="205" fill="url(#grid)"/>

          {/* Power flow lines */}
          {[0, 1, 2, 3].map(i => (
            <FlowLine key={i} idx={i}
              isActive={pos !== null && (pos === i || pos === i + 1)}
              condColor={COND_COLOR[cond]}/>
          ))}

          {/* Sector nodes */}
          {SECTORS.map((_, i) => (
            <SectorNode key={i} i={i} pos={pos} cond={cond}/>
          ))}

          {/* "No data" overlay */}
          {!step && (
            <text x="500" y="105"
              textAnchor="middle" dominantBaseline="middle"
              fill="#162840" fontSize="13" fontWeight="600" letterSpacing="2">
              INICIA LA SIMULACIÓN PARA ACTIVAR LA RED
            </text>
          )}
        </svg>
      </div>

      {/* Capacity bar + stats */}
      <div style={{
        display: "grid", gridTemplateColumns: "1fr auto",
        gap: "1rem", alignItems: "center", marginTop: ".85rem",
      }}>
        <div>
          <div style={{
            display: "flex", justifyContent: "space-between",
            marginBottom: ".35rem",
          }}>
            <span style={{ fontSize: ".65rem", color: "#475569", textTransform: "uppercase", letterSpacing: ".08em", fontWeight: 700 }}>
              Capacidad de generación
            </span>
            <span style={{ fontSize: ".65rem", color: capColor, fontWeight: 800 }}>
              {capLabel} · {res}/10 MW
            </span>
          </div>
          <div style={{
            height: 8, borderRadius: 4,
            background: "#0b1a2e",
            border: "1px solid #162840", overflow: "hidden",
          }}>
            <div style={{
              height: "100%", width: `${capPct}%`,
              borderRadius: 4, background: capColor,
              transition: "width .4s cubic-bezier(.4,0,.2,1), background .3s",
              boxShadow: `0 0 8px ${capColor}66`,
            }}/>
          </div>
        </div>

        {/* Position indicator */}
        <div style={{ display: "flex", gap: 5 }}>
          {SECTORS.map((s, i) => (
            <div key={i} style={{
              width: 28, height: 28, borderRadius: 6,
              background: pos === i ? COND_COLOR[cond] + "22" : "#0b1a2e",
              border: `1.5px solid ${pos === i ? COND_COLOR[cond] : "#162840"}`,
              display: "flex", alignItems: "center", justifyContent: "center",
              fontSize: pos === i ? ".48rem" : ".6rem",
              fontWeight: 800, fontFamily: "monospace",
              color: pos === i ? COND_COLOR[cond] : "#2a4060",
              transition: "all .3s",
              boxShadow: pos === i ? `0 0 10px ${COND_COLOR[cond]}44` : "none",
            }}>
              {pos === i ? s.abbr : s.short}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
