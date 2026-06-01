<template>
    <UDashboardGroup>
        <UDashboardSidebar resizable :default-size="20" :min-size="15" :max-size="25" class="border-r border-default">
            <Sidebar />
        </UDashboardSidebar>

        <UDashboardPanel grow>
            <div class="h-[calc(100vh)] w-full flex">
                <Mainpanel />
            </div>
        </UDashboardPanel>

        <UDashboardSidebar side="right" resizable :default-size="25" :min-size="20" :max-size="50" class="border-l border-default" :ui="{ body: 'px-0 py-0' }">
            <MapPanel class="px-0 py-0 gap-0" />
        </UDashboardSidebar>

    </UDashboardGroup>
</template>

<script setup lang="ts">
import type { DropdownMenuItem } from '@nuxt/ui'

const items = ref<DropdownMenuItem[]>([
    {
        label: 'Auto Camera Track',
        icon: 'i-lucide-eye',
        onClick: () => {
            console.log('Auto Camera Track clicked')
        }
    },
])
</script>