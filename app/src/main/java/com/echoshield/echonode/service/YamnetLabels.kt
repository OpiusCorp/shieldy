package com.echoshield.echonode.service

/** Firearm evidence only; generic impulses must not raise firearm confidence. */
internal object YamnetLabels {
    private val firearmLabels = setOf(
        "Gunshot, gunfire", "Machine gun", "Fusillade", "Artillery fire"
    )

    fun isGunshot(label: String): Boolean = label in firearmLabels
}
