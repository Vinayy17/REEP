import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"

const cn = (...classes) => classes.filter(Boolean).join(" ")

const textToneClasses = {
  default: "text-white",
  success: "text-emerald-300",
  warning: "text-amber-300",
  danger: "text-rose-300",
  info: "text-indigo-300",
  muted: "text-slate-400",
}

const statusToneClasses = {
  default: "border-white/10 bg-white/5 text-slate-200",
  success: "border-emerald-400/20 bg-emerald-500/15 text-emerald-200",
  warning: "border-amber-400/20 bg-amber-500/15 text-amber-200",
  danger: "border-rose-400/20 bg-rose-500/15 text-rose-200",
  info: "border-indigo-400/20 bg-indigo-500/15 text-indigo-200",
}

const actionToneClasses = {
  primary:
    "border-indigo-400/20 bg-indigo-500 text-white hover:bg-indigo-400 hover:text-white",
  secondary:
    "border-white/10 bg-transparent text-white hover:bg-white/5 hover:text-white",
  success:
    "border-emerald-400/20 bg-emerald-500/10 text-emerald-200 hover:bg-emerald-500/20 hover:text-white",
  danger:
    "border-rose-400/20 bg-rose-500/10 text-rose-200 hover:bg-rose-500/20 hover:text-white",
}

export function EntityActionRow({ actions = [], compact = false }) {
  if (!actions.length) return null

  return (
    <div className={cn("flex flex-wrap gap-2 border-t border-white/10", compact ? "pt-2.5" : "pt-3")}>
      {actions.map((action) => {
        const Icon = action.icon
        return (
          <Button
            key={action.key || action.label}
            type="button"
            variant="outline"
            className={cn(
              compact
                ? "h-9 min-w-[84px] flex-1 rounded-xl px-2.5 text-[11px] font-medium shadow-none"
                : "h-10 min-w-[92px] flex-1 rounded-xl px-3 text-xs font-medium shadow-none",
              actionToneClasses[action.tone || "secondary"]
            )}
            onClick={action.onClick}
            disabled={action.disabled}
          >
            {Icon ? <Icon className={cn("mr-1.5", compact ? "h-3.5 w-3.5" : "h-4 w-4")} /> : null}
            {action.label}
          </Button>
        )
      })}
    </div>
  )
}

export default function EntityCard({
  icon,
  title,
  subtitle,
  metaLines = [],
  amount,
  amountTone = "default",
  status,
  breakdown = [],
  actions = [],
  onOpen,
  className = "",
  compact = false,
}) {
  const columnCount = breakdown.length > 3 ? 2 : Math.max(breakdown.length, 1)

  return (
    <Card
      className={cn(
        "rounded-[20px] border border-white/8 bg-[linear-gradient(180deg,rgba(255,255,255,0.035),rgba(255,255,255,0.016))] text-white shadow-[0_18px_40px_rgba(2,6,23,0.12)] backdrop-blur-xl",
        className
      )}
    >
      <CardContent className={cn(compact ? "space-y-3 p-3" : "space-y-4 p-4")}>
        <div
          className={cn(compact ? "space-y-2.5" : "space-y-3", onOpen ? "cursor-pointer" : "")}
          onClick={onOpen}
        >
          <div className={cn("flex items-start justify-between", compact ? "gap-3" : "gap-4")}>
            <div className={cn("flex min-w-0 flex-1 items-start", compact ? "gap-2.5" : "gap-3")}>
              <div className={cn(
                "flex shrink-0 items-center justify-center rounded-[18px] bg-white/[0.045] text-indigo-300",
                compact ? "h-10 w-10" : "h-11 w-11"
              )}>
                {icon}
              </div>

              <div className={cn("min-w-0", compact ? "space-y-0.5" : "space-y-1")}>
                <p className={cn(
                  "truncate font-semibold leading-none text-white",
                  compact ? "text-[15px]" : "text-lg"
                )}>
                  {title}
                </p>
                {subtitle ? (
                  <p className={cn("truncate text-slate-400", compact ? "text-[13px]" : "text-sm")}>{subtitle}</p>
                ) : null}
                {metaLines
                  .filter(Boolean)
                  .map((line, index) => (
                    <p
                      key={`${title}-${index}`}
                      className={cn(
                        "text-slate-400",
                        compact ? "text-[13px] leading-4.5" : "text-sm leading-5"
                      )}
                    >
                      {line}
                    </p>
                  ))}
              </div>
            </div>

            <div className={cn("shrink-0 text-right", compact ? "space-y-1.5 pl-1" : "space-y-2 pl-2")}>
              {amount ? (
                <p className={cn(
                  compact ? "text-base font-semibold leading-none" : "text-lg font-semibold leading-none",
                  textToneClasses[amountTone] || textToneClasses.default
                )}>
                  {amount}
                </p>
              ) : null}
              {status?.label ? (
                <span
                  className={cn(
                    compact
                      ? "inline-flex rounded-full border px-2.5 py-0.5 text-[10px] font-semibold"
                      : "inline-flex rounded-full border px-3 py-1 text-[11px] font-semibold",
                    statusToneClasses[status.tone || "default"]
                  )}
                >
                  {status.label}
                </span>
              ) : null}
            </div>
          </div>

          {breakdown.length ? (
            <div
              className={cn(
                "grid rounded-[18px] border border-white/8 bg-white/[0.025]",
                compact ? "gap-2.5 p-2.5" : "gap-3 p-3"
              )}
              style={{ gridTemplateColumns: `repeat(${columnCount}, minmax(0, 1fr))` }}
            >
              {breakdown.map((item) => (
                <div key={item.key || item.label} className={cn(compact ? "space-y-0.5" : "space-y-1")}>
                  <p className={cn(
                    "uppercase text-slate-400",
                    compact ? "text-[10px] tracking-[0.18em]" : "text-[11px] tracking-[0.22em]"
                  )}>
                    {item.label}
                  </p>
                  <p className={cn(
                    compact ? "text-[13px] font-semibold" : "text-sm font-semibold",
                    textToneClasses[item.tone] || textToneClasses.default
                  )}>
                    {item.value}
                  </p>
                </div>
              ))}
            </div>
          ) : null}
        </div>

        <EntityActionRow actions={actions} compact={compact} />
      </CardContent>
    </Card>
  )
}
