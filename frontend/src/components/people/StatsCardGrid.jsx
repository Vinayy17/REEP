import { Card, CardContent } from "@/components/ui/card"

const cn = (...classes) => classes.filter(Boolean).join(" ")

const toneClasses = {
  default: "text-white",
  success: "text-emerald-300",
  warning: "text-amber-300",
  danger: "text-rose-300",
  info: "text-indigo-300",
}

export default function StatsCardGrid({ items = [], className = "" }) {
  if (!items.length) return null

  return (
    <div className={cn("grid grid-cols-2 gap-3 lg:grid-cols-4", className)}>
      {items.map((item) => {
        const tone = toneClasses[item.tone] || toneClasses.default
        const indicatorTone = item.indicatorTone ? toneClasses[item.indicatorTone] || tone : null

        return (
          <Card
            key={item.key || item.label}
            className="rounded-[18px] border border-white/8 bg-[linear-gradient(180deg,rgba(255,255,255,0.032),rgba(255,255,255,0.014))] text-white shadow-[0_14px_30px_rgba(2,6,23,0.1)] backdrop-blur-xl"
          >
            <CardContent className="space-y-1.5 p-3 sm:p-4">
              <div className="flex items-center justify-between gap-2">
                <p className="text-[10px] uppercase tracking-[0.22em] text-slate-400 sm:text-[11px]">
                  {item.label}
                </p>
                {indicatorTone ? <span className={cn("h-2.5 w-2.5 rounded-full", indicatorTone.replace("text-", "bg-"))} /> : null}
              </div>
              <p className={cn("text-sm font-semibold leading-tight sm:text-xl", tone)}>
                {item.value}
              </p>
            </CardContent>
          </Card>
        )
      })}
    </div>
  )
}
