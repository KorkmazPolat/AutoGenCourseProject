
export enum UserRole {
    Admin = 'Admin',
    CourseCreator = 'CourseCreator',
    Student = 'Student',
}

export interface TelemetryData {
    feedback: any;
    performance: any;
    workload: any;
}

export interface MappedTelemetry {
    // Admin View
    errorGraphs?: any;
    performanceHeatmaps?: any;
    technicalQueue?: any;

    // Creator View
    progressBar?: {
        percent: number;
        status: string;
    };
    estimatedWaitTime?: string;
    smartAlerts?: string[];
}

export class TelemetryMapper {
    static map(rawTelemetry: TelemetryData, role: UserRole): MappedTelemetry | null {
        if (role === UserRole.Student) {
            return null;
        }

        if (role === UserRole.Admin) {
            return this.mapForAdmin(rawTelemetry);
        }

        if (role === UserRole.CourseCreator) {
            return this.mapForCreator(rawTelemetry);
        }

        return null;
    }

    private static mapForAdmin(data: TelemetryData): MappedTelemetry {
        return {
            errorGraphs: data.feedback.failure_trends,
            performanceHeatmaps: data.performance,
            technicalQueue: {
                activeTasks: data.workload.active_tasks,
                queueDepth: data.workload.max_queue_depth, // or current depth if available in summary
                taskLog: data.workload.task_log,
            },
            // Admin also gets creator view data usually, but per requirements "Must see..."
            // We can include creator data if useful, but requirements specify distinct views.
            // I'll stick to what's requested.
        };
    }

    private static mapForCreator(data: TelemetryData): MappedTelemetry {
        // 1. Simplified Progress Bar
        // We can estimate progress based on completed stages vs expected.
        // Or use workload tasks.
        // Let's use feedback events to count completed lessons/stages.
        const totalEvents = data.feedback.total_events;
        // A rough heuristic: if we know total expected lessons, we could calculate %.
        // Without that, we can just show "Working..." or use active tasks.
        const activeTasks = data.workload.active_tasks;
        const percent = activeTasks > 0 ? 50 : 100; // Placeholder logic
        const status = activeTasks > 0 ? `Processing ${activeTasks} tasks...` : 'Ready';

        // 2. Estimated Wait Time
        // Simple heuristic: 2 minutes per active task
        const waitTimeMinutes = activeTasks * 2;
        const estimatedWaitTime = activeTasks > 0 ? `~${waitTimeMinutes} mins` : 'Ready';

        // 3. Smart Alerts
        // Check for critical failures in feedback
        const alerts: string[] = [];
        const recentFailures = data.feedback.recent_failures || [];
        for (const failure of recentFailures) {
            // specific logic for creator-friendly messages
            if (failure.stage.includes('review')) {
                alerts.push("Please refine your learning outcomes to improve content quality.");
            }
        }
        // Deduplicate
        const uniqueAlerts = Array.from(new Set(alerts));

        return {
            progressBar: {
                percent: percent,
                status: status,
            },
            estimatedWaitTime: estimatedWaitTime,
            smartAlerts: uniqueAlerts.length > 0 ? uniqueAlerts : undefined,
        };
    }
}
