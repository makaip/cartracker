<template>
    <UDashboardGroup>
        <!-- Sidebar -->
        <UDashboardSidebar resizable :default-size="20" :min-size="15" :max-size="50" class="border-r border-default">
            <Sidebar />
        </UDashboardSidebar>

        <!-- Main -->
        <UDashboardPanel grow>
            <div class="h-[calc(100vh)] w-full flex">
                <Mainpanel />
            </div>
        </UDashboardPanel>

    </UDashboardGroup>


    <!-- modal   
  -->


</template>

<script setup lang="ts">
const Placeholder = defineComponent({
    name: 'Placeholder',
    template: `
    <div class="text-sm text-muted">
      Placeholder
    </div>
  `
})

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